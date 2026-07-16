#!/usr/bin/env python3
"""Fast daily train: beat_v3 + large-chunk augmentation + live-shaped top-K eval.

Beats xgb_v3_holdout on sealed holdout. Used by daily_refresh_retrain.

Live validators send ~80–100 hands/chunk while public benchmark is ~30–40.
This recipe merges same-label chunks into 50–100 hand views for training and
reports top-K reward at batch sizes 40 and 100.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import beat_v3_schema as schema
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
from poker44.validator.payload_view import prepare_hand_for_miner
from scripts.train.train_competitive_v3 import (
    apply_post,
    eval_suite,
    lodo_oof,
    recency_weights,
    selection_score,
    simulate_batch_reward,
    tune_cal_and_logit,
)

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT = ROOT / "models" / "competitive"


def make_xgb(seed: int) -> XGBClassifier:
    """v3ish capacity config that beat xgb_v3_holdout on sealed holdout."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        reg_alpha=0.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )


def load_raw_examples(path: Path):
    chunks: list[list[dict]] = []
    ys: list[int] = []
    dates: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            ex = json.loads(line)
            hands = [
                prepare_hand_for_miner(h)
                for h in (ex.get("hands") or [])
                if isinstance(h, dict)
            ]
            if not hands:
                continue
            chunks.append(hands)
            ys.append(int(ex["label"]))
            dates.append(str(ex["sourceDate"]))
            if (i + 1) % 500 == 0:
                print(f"  labeled {i + 1}", flush=True)
    return chunks, np.asarray(ys, dtype=np.int64), np.asarray(dates)


def featurize_chunks(chunks: list[list[dict]]) -> np.ndarray:
    names = list(schema.FEATURE_NAMES)
    rows = []
    for i, hands in enumerate(chunks):
        feat = schema.extract_chunk_features(hands)
        rows.append(schema.features_to_vector(feat))
        if (i + 1) % 500 == 0:
            print(f"  featurized {i + 1}/{len(chunks)}", flush=True)
    return np.asarray(rows, dtype=np.float64), names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--recent-val-days", type=int, default=4)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-large-aug", action="store_true", help="Disable 50–100 hand merges")
    ap.add_argument("--large-ratio", type=float, default=1.25)
    ap.add_argument("--medium-ratio", type=float, default=0.40)
    ap.add_argument("--archive", action="store_true", help="Copy current.joblib to archive/ before overwrite")
    args = ap.parse_args()

    if not args.examples.exists():
        raise SystemExit(f"Missing examples: {args.examples}")

    names = list(schema.FEATURE_NAMES)
    print(f"Loading raw chunks from {args.examples}...", flush=True)
    chunks, y, dates = load_raw_examples(args.examples)
    unique = sorted(set(dates.tolist()))
    holdout_dates = unique[-args.holdout_days :]
    recent_dates = unique[-(args.holdout_days + args.recent_val_days) : -args.holdout_days]
    hold_set = set(holdout_dates)
    pool_mask = np.array([d not in hold_set for d in dates])
    hold_mask = ~pool_mask
    print(
        f"n={len(y)} dates={unique[0]}..{unique[-1]} "
        f"holdout={holdout_dates} lodo={recent_dates} pool={pool_mask.sum()} hold={hold_mask.sum()}",
        flush=True,
    )

    aug_cfg = LargeAugmentationConfig(
        large_ratio=0.0 if args.no_large_aug else float(args.large_ratio),
        medium_ratio=0.0 if args.no_large_aug else float(args.medium_ratio),
    )
    pool_idx = np.flatnonzero(pool_mask)
    hold_idx = np.flatnonzero(hold_mask)
    pool_chunks = [chunks[i] for i in pool_idx]
    pool_y = y[pool_mask]
    pool_dates = dates[pool_mask]

    print(
        f"Augmenting pool (large_ratio={aug_cfg.large_ratio}, medium_ratio={aug_cfg.medium_ratio})...",
        flush=True,
    )
    if args.no_large_aug:
        train_chunks = pool_chunks
        train_y = pool_y
        train_dates = [str(d) for d in pool_dates.tolist()]
        aug_stats = {"original": len(pool_chunks), "large": 0, "medium": 0, "full_total": len(pool_chunks)}
    else:
        train_chunks, train_y, train_dates, aug_stats = build_training_views(
            pool_chunks,
            pool_y,
            pool_dates.tolist(),
            aug_cfg,
            seed=args.seed,
        )
    print(f"  aug stats={aug_stats}", flush=True)

    hold_chunks = [chunks[i] for i in hold_idx]
    hold_y = y[hold_mask]
    hold_dates = [str(d) for d in dates[hold_mask].tolist()]

    print("Featurizing train+hold...", flush=True)
    X_train, names = featurize_chunks(train_chunks)
    X_hold, _ = featurize_chunks(hold_chunks)
    y_train = np.asarray(train_y, dtype=np.int64)
    dates_train = np.asarray(train_dates)
    # LODO pool_mask is all-True for the train matrix (holdout excluded already).
    train_pool_mask = np.ones(len(y_train), dtype=bool)

    print("LODO + calibrate...", flush=True)
    oof, oy = lodo_oof(
        X_train,
        y_train,
        dates_train,
        train_pool_mask,
        recent_dates,
        X_hn=None,
        y_hn=None,
        hn_weight=0.0,
        half_life=args.half_life_days,
        seed=args.seed,
        model_factory=make_xgb,
    )
    cal, logit, _ = tune_cal_and_logit(oy, oof, args.seed)
    scored = apply_post(oof, calibrator=cal, logit=logit)
    m_lodo = eval_suite(oy, scored, "lodo")
    print(
        f"  LODO reward={m_lodo['reward']:.4f} ap={m_lodo['ap']:.4f} "
        f"bot@5fpr={m_lodo['bot_recall_at_5fpr']:.4f} logit={logit}",
        flush=True,
    )

    print("Sealed holdout (original chunk size)...", flush=True)
    w_pool = recency_weights(
        dates_train, sorted(set(dates_train.tolist())), args.half_life_days
    )
    model_pool = make_xgb(args.seed)
    model_pool.fit(X_train, y_train, sample_weight=w_pool)
    ho = apply_post(model_pool.predict_proba(X_hold)[:, 1], calibrator=cal, logit=logit)
    m_ho = eval_suite(hold_y, ho, "holdout_sealed")
    m_topk40 = simulate_batch_reward(hold_y, ho, batch_size=40, fraction=0.125)
    m_topk100 = simulate_batch_reward(hold_y, ho, batch_size=100, fraction=0.125)
    print(
        f"  SEALED reward={m_ho['reward']:.4f} ap={m_ho['ap']:.4f} "
        f"bot@5fpr={m_ho['bot_recall_at_5fpr']:.4f} "
        f"topk40={m_topk40['reward']:.4f} topk100={m_topk100['reward']:.4f}",
        flush=True,
    )

    m_live = None
    m_live_topk100 = None
    if not args.no_large_aug and len(hold_chunks) >= 6:
        print("Live-shaped sealed holdout (80–100 hands)...", flush=True)
        live_chunks, live_y, live_dates = build_live_shaped_holdout(
            hold_chunks,
            hold_y,
            hold_dates,
            seed=args.seed + 7,
            ratio=1.0,
        )
        X_live, _ = featurize_chunks(live_chunks)
        live_scores = apply_post(
            model_pool.predict_proba(X_live)[:, 1], calibrator=cal, logit=logit
        )
        m_live = eval_suite(live_y, live_scores, "holdout_live_shaped")
        m_live_topk100 = simulate_batch_reward(
            live_y, live_scores, batch_size=100, fraction=0.125
        )
        print(
            f"  LIVE-SHAPED reward={m_live['reward']:.4f} ap={m_live['ap']:.4f} "
            f"bot@5fpr={m_live['bot_recall_at_5fpr']:.4f} "
            f"topk100={m_live_topk100['reward']:.4f} n={len(live_y)}",
            flush=True,
        )

    print("Deploy fit on all labeled (+ aug)...", flush=True)
    if args.no_large_aug:
        deploy_chunks = chunks
        deploy_y = y
        deploy_dates = [str(d) for d in dates.tolist()]
        deploy_aug_stats = {"original": len(chunks), "large": 0, "medium": 0, "full_total": len(chunks)}
    else:
        deploy_chunks, deploy_y, deploy_dates, deploy_aug_stats = build_training_views(
            chunks,
            y,
            dates.tolist(),
            aug_cfg,
            seed=args.seed + 99,
        )
    print(f"  deploy aug stats={deploy_aug_stats}", flush=True)
    X_deploy, _ = featurize_chunks(deploy_chunks)
    y_deploy = np.asarray(deploy_y, dtype=np.int64)
    dates_deploy = np.asarray(deploy_dates)
    w_all = recency_weights(dates_deploy, sorted(set(dates_deploy.tolist())), args.half_life_days)
    model = make_xgb(args.seed)
    model.fit(X_deploy, y_deploy, sample_weight=w_all)

    trained_at = datetime.now(timezone.utc).isoformat()
    fingerprint = recipe_fingerprint(ROOT)
    report = {
        "trained_at_utc": trained_at,
        "model_name": "poker44-beat-v3",
        "model_version": "7.1.0",
        "n_features": len(names),
        "feature_set": "beat_v3",
        "recipe_fingerprint": fingerprint,
        "half_life_days": args.half_life_days,
        "score_remap": logit,
        "holdout_dates": holdout_dates,
        "recent_lodo_dates": recent_dates,
        "latest_source_date": unique[-1],
        "benchmark_n": int(len(y)),
        "augmentation": {
            "enabled": not args.no_large_aug,
            "config": aug_cfg.as_dict(),
            "pool_stats": aug_stats,
            "deploy_stats": deploy_aug_stats,
        },
        "label_counts": {
            "all": dict(Counter(y.tolist())),
            "holdout": dict(Counter(hold_y.tolist())),
            "train_with_aug": dict(Counter(y_train.tolist())),
        },
        "metrics": {
            "lodo": m_lodo,
            "lodo_selection_score": selection_score(m_lodo),
            "holdout_sealed": m_ho,
            "holdout_sealed_topk40": m_topk40,
            "holdout_sealed_topk100": m_topk100,
            "holdout_live_shaped": m_live,
            "holdout_live_shaped_topk100": m_live_topk100,
        },
        "selection_policy": (
            "daily recipe: beat_v3 + large-chunk aug (80-100 hands) + LODO; "
            "holdout sealed; topk eval at bs=40 and bs=100; deploy all+aug"
        ),
    }

    metadata = {
        "model_name": report["model_name"],
        "model_version": report["model_version"],
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
        "score_remap": logit or {},
        "feature_set": "beat_v3",
        "holdout_dates": holdout_dates,
        "latest_source_date": unique[-1],
        "trained_at_utc": trained_at,
        "framework": "xgb_beat_v3_large",
        "recipe_fingerprint": fingerprint,
        "augmentation": report["augmentation"],
    }
    artifact = {
        "kind": "single",
        "models": [model],
        "weights": [1.0],
        "calibrator": cal,
        "feature_names": names,
        "feature_set": "beat_v3",
        "metadata": metadata,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "current.joblib"
    if args.archive and out_path.exists():
        arch = args.out_dir / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(out_path, arch / f"current_{stamp}.joblib")
        prev_report = args.out_dir / "train_report.json"
        if prev_report.exists():
            shutil.copy2(prev_report, arch / f"train_report_{stamp}.json")
        prune_archive(arch, keep=int(os.getenv("POKER44_ARCHIVE_KEEP", "30")))
        print(f"Archived previous artifact under {arch}", flush=True)

    atomic_joblib_dump(artifact, out_path)
    atomic_write_text(args.out_dir / "train_report.json", json.dumps(report, indent=2) + "\n")
    atomic_write_text(
        args.out_dir / "threshold.json",
        json.dumps(
            {
                "threshold": 0.5,
                "batch_calibration": "topk_v1",
                "score_remap": logit,
                "model_name": report["model_name"],
                "model_version": report["model_version"],
                "latest_source_date": unique[-1],
                "recipe_fingerprint": fingerprint,
                "model_path": str(out_path),
            },
            indent=2,
        )
        + "\n",
    )
    print(json.dumps({"latest_source_date": unique[-1], "metrics": report["metrics"]}, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
