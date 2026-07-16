#!/usr/bin/env python3
"""Diverse ensemble (XGB + ExtraTrees + Logistic) fused by within-request rank.

Deploys ONLY if the candidate beats the currently-deployed model on a live-shaped
holdout (80-100 hand chunks, top-K at batch size 100), evaluated through the exact
serving path (CompetitiveMinerModel) for both candidate and current.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import beat_v3_coherent_schema as schema
from poker44.artifact_io import (
    atomic_joblib_dump,
    atomic_write_text,
    prune_archive,
    recipe_fingerprint,
)
from poker44.large_chunk_augment import (
    LargeAugmentationConfig,
    build_live_shaped_holdout,
    build_training_views,
)
from poker44.miner_inference import CompetitiveMinerModel
from scripts.train.train_beat_v3_coherent import featurize, load_raw_examples
from scripts.train.train_competitive_v3 import eval_suite, recency_weights

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT = ROOT / "models" / "competitive"

# Fixed priors (no search on the gate holdout => no selection leakage).
BRANCH_WEIGHTS = {"xgb": 0.60, "extratrees": 0.25, "logistic": 0.15}
MIN_RANK_BATCH = 8


def make_xgb(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.7,
        min_child_weight=2,
        reg_lambda=2.5,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )


def make_extratrees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=4,
        random_state=seed,
    )


def make_logistic(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)


def build_branches(X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int) -> list[dict]:
    xgb = make_xgb(seed)
    xgb.fit(X, y, sample_weight=w)
    et = make_extratrees(seed)
    et.fit(X, y, sample_weight=w)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = make_logistic(seed)
    lr.fit(Xs, y, sample_weight=w)
    return [
        {"name": "xgb", "model": xgb, "weight": BRANCH_WEIGHTS["xgb"], "scaler": None},
        {"name": "extratrees", "model": et, "weight": BRANCH_WEIGHTS["extratrees"], "scaler": None},
        {"name": "logistic", "model": lr, "weight": BRANCH_WEIGHTS["logistic"], "scaler": scaler},
    ]


def make_artifact(branches: list[dict], names: list[str], metadata: dict) -> dict:
    return {
        "kind": "rank_blend_v1",
        "branches": branches,
        "feature_names": names,
        "feature_set": "beat_v3_coherent",
        "min_rank_batch": MIN_RANK_BATCH,
        "metadata": metadata,
    }


def base_metadata(names: list[str], unique: list[str], holdout_dates, fingerprint, trained_at) -> dict:
    return {
        "model_name": "poker44-rank-ensemble",
        "model_version": "9.0.0",
        "decision_threshold": 0.5,
        "batch_calibration_default": "topk_v1",
        "batch_safety_budget": {
            "kind": "topk_v1",
            "max_positive_fraction": 0.125,
            "max_positive_count": 40,
            "positive_floor": 0.501,
            "positive_ceiling": 0.509,
            "negative_ceiling": 0.49,
        },
        "score_remap": {},
        "feature_set": "beat_v3_coherent",
        "min_rank_batch": MIN_RANK_BATCH,
        "holdout_dates": holdout_dates,
        "latest_source_date": unique[-1],
        "trained_at_utc": trained_at,
        "framework": "rank_ensemble_beat_v3_coherent",
        "recipe_fingerprint": fingerprint,
    }


def score_model_windows(model: CompetitiveMinerModel, chunks, y, *, window=100) -> dict:
    final = np.zeros(len(chunks), dtype=float)
    for start in range(0, len(chunks), window):
        sl = slice(start, start + window)
        final[sl] = np.asarray(model.score_chunks(chunks[sl]), dtype=float)
    return eval_suite(y, final, f"windows_bs{window}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--large-ratio", type=float, default=1.25)
    ap.add_argument("--medium-ratio", type=float, default=0.40)
    ap.add_argument("--min-gain", type=float, default=0.003)
    ap.add_argument("--force-deploy", action="store_true")
    args = ap.parse_args()

    if not args.examples.exists():
        raise SystemExit(f"Missing examples: {args.examples}")

    names = list(schema.FEATURE_NAMES)
    print(f"beat_v3_coherent features = {len(names)}", flush=True)
    chunks, y, dates = load_raw_examples(args.examples)
    unique = sorted(set(dates.tolist()))
    holdout_dates = unique[-args.holdout_days :]
    hold_set = set(holdout_dates)
    pool_mask = np.array([d not in hold_set for d in dates])
    hold_mask = ~pool_mask
    print(f"n={len(y)} holdout={holdout_dates} pool={pool_mask.sum()} hold={hold_mask.sum()}", flush=True)

    aug_cfg = LargeAugmentationConfig(large_ratio=args.large_ratio, medium_ratio=args.medium_ratio)
    pool_idx = np.flatnonzero(pool_mask)
    pool_chunks = [chunks[i] for i in pool_idx]
    pool_y = y[pool_mask]
    pool_dates = dates[pool_mask]
    hold_idx = np.flatnonzero(hold_mask)
    hold_chunks = [chunks[i] for i in hold_idx]
    hold_y = y[hold_mask]
    hold_dates = [str(d) for d in dates[hold_mask].tolist()]

    print("Augment pool + featurize...", flush=True)
    train_chunks, train_y, train_dates, aug_stats = build_training_views(
        pool_chunks, pool_y, pool_dates.tolist(), aug_cfg, seed=args.seed
    )
    print(f"  aug stats={aug_stats}", flush=True)
    X_train = featurize(train_chunks)
    y_train = np.asarray(train_y, dtype=np.int64)
    dates_train = np.asarray(train_dates)
    w_pool = recency_weights(dates_train, sorted(set(dates_train.tolist())), args.half_life_days)

    print("Training candidate branches (pool)...", flush=True)
    branches_pool = build_branches(X_train, y_train, w_pool, args.seed)

    trained_at = datetime.now(timezone.utc).isoformat()
    fingerprint = recipe_fingerprint(ROOT)
    metadata = base_metadata(names, unique, holdout_dates, fingerprint, trained_at)

    print("Build live-shaped holdout + evaluate via serving path...", flush=True)
    live_chunks, live_y, _ = build_live_shaped_holdout(
        hold_chunks, hold_y, hold_dates, seed=args.seed + 7, ratio=1.0
    )
    print(f"  live-shaped n={len(live_y)}", flush=True)

    with tempfile.TemporaryDirectory() as td:
        cand_path = Path(td) / "cand.joblib"
        atomic_joblib_dump(make_artifact(branches_pool, names, metadata), cand_path)
        cand_model = CompetitiveMinerModel(model_path=cand_path)
        cand_metric = score_model_windows(cand_model, live_chunks, live_y, window=100)
    print(
        f"  CANDIDATE(rank-ensemble) live reward={cand_metric['reward']:.4f} "
        f"ap={cand_metric['ap']:.4f} bot@5fpr={cand_metric['bot_recall_at_5fpr']:.4f}",
        flush=True,
    )

    current_path = args.out_dir / "current.joblib"
    current_metric = None
    current_version = None
    if current_path.exists():
        cur_model = CompetitiveMinerModel(model_path=current_path)
        current_version = cur_model.model_version
        current_metric = score_model_windows(cur_model, live_chunks, live_y, window=100)
        print(
            f"  CURRENT({current_version}) live reward={current_metric['reward']:.4f} "
            f"ap={current_metric['ap']:.4f} bot@5fpr={current_metric['bot_recall_at_5fpr']:.4f}",
            flush=True,
        )

    cand_gate = cand_metric["reward"]
    cur_gate = current_metric["reward"] if current_metric else -1.0
    gain = cand_gate - cur_gate
    beat = (current_metric is None) or (gain >= args.min_gain) or args.force_deploy
    print(
        f"\nGATE: candidate={cand_gate:.4f} vs current={cur_gate:.4f} gain={gain:+.4f} "
        f"min_gain={args.min_gain:.4f} -> {'DEPLOY' if beat else 'HOLD (keep current)'}",
        flush=True,
    )

    report = {
        "trained_at_utc": trained_at,
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "n_features": len(names),
        "feature_set": "beat_v3_coherent",
        "recipe_fingerprint": fingerprint,
        "latest_source_date": unique[-1],
        "holdout_dates": holdout_dates,
        "branch_weights": BRANCH_WEIGHTS,
        "min_rank_batch": MIN_RANK_BATCH,
        "augmentation": {"config": aug_cfg.as_dict(), "pool_stats": aug_stats},
        "gate": {
            "candidate_live": cand_gate,
            "current_live": cur_gate,
            "current_version": current_version,
            "gain": gain,
            "min_gain": args.min_gain,
            "deployed": bool(beat),
            "forced": bool(args.force_deploy),
        },
        "metrics": {"candidate_live_shaped": cand_metric, "current_live_shaped": current_metric},
        "selection_policy": (
            "rank ensemble (xgb+extratrees+logistic) fused by within-request percentile; "
            "deploy ONLY if candidate beats current live model on live-shaped topk100"
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not beat:
        cand_out = args.out_dir / "candidate_rank_ensemble.joblib"
        atomic_joblib_dump(make_artifact(branches_pool, names, metadata), cand_out)
        atomic_write_text(args.out_dir / "candidate_report.json", json.dumps(report, indent=2) + "\n")
        print(f"HELD: saved aside -> {cand_out}. Live model untouched.", flush=True)
        print(json.dumps({"gate": report["gate"]}, indent=2))
        return 0

    print("Refit branches on all+aug for deploy...", flush=True)
    deploy_chunks, deploy_y, deploy_dates, deploy_aug = build_training_views(
        chunks, y, dates.tolist(), aug_cfg, seed=args.seed + 99
    )
    X_deploy = featurize(deploy_chunks)
    y_deploy = np.asarray(deploy_y, dtype=np.int64)
    dates_deploy = np.asarray(deploy_dates)
    w_all = recency_weights(dates_deploy, sorted(set(dates_deploy.tolist())), args.half_life_days)
    branches_all = build_branches(X_deploy, y_deploy, w_all, args.seed)
    report["augmentation"]["deploy_stats"] = deploy_aug

    if current_path.exists():
        arch = args.out_dir / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(current_path, arch / f"current_{stamp}.joblib")
        prev_report = args.out_dir / "train_report.json"
        if prev_report.exists():
            shutil.copy2(prev_report, arch / f"train_report_{stamp}.json")
        prune_archive(arch, keep=int(os.getenv("POKER44_ARCHIVE_KEEP", "30")))
        print(f"Archived previous artifact under {arch}", flush=True)

    atomic_joblib_dump(make_artifact(branches_all, names, metadata), current_path)
    atomic_write_text(args.out_dir / "train_report.json", json.dumps(report, indent=2) + "\n")
    atomic_write_text(
        args.out_dir / "threshold.json",
        json.dumps(
            {
                "threshold": 0.5,
                "batch_calibration": "topk_v1",
                "score_remap": {},
                "model_name": metadata["model_name"],
                "model_version": metadata["model_version"],
                "latest_source_date": unique[-1],
                "recipe_fingerprint": fingerprint,
                "model_path": str(current_path),
            },
            indent=2,
        )
        + "\n",
    )
    print(f"DEPLOYED rank-ensemble -> {current_path}", flush=True)
    print(json.dumps({"gate": report["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
