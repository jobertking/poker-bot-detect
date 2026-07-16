#!/usr/bin/env python3
"""Train beat_v3 + coherent candidate, deploy ONLY if it beats the live model.

Gate: candidate must beat the currently-deployed current.joblib on a live-shaped
holdout (80-100 hand chunks, top-K at batch size 100) by >= --min-gain reward,
otherwise the live artifact is left untouched and the candidate is saved aside.
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
    return XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.7,
        min_child_weight=2,
        reg_lambda=2.5,
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
                print(f"  loaded {i + 1}", flush=True)
    return chunks, np.asarray(ys, dtype=np.int64), np.asarray(dates)


def featurize(chunks: list[list[dict]]) -> np.ndarray:
    names = list(schema.FEATURE_NAMES)
    rows = []
    for i, hands in enumerate(chunks):
        feat = schema.extract_chunk_features(hands)
        rows.append([float(feat.get(n, 0.0)) for n in names])
        if (i + 1) % 500 == 0:
            print(f"  featurized {i + 1}/{len(chunks)}", flush=True)
    return np.asarray(rows, dtype=np.float64)


def score_live_model_windows(
    model: CompetitiveMinerModel, chunks: list[list[dict]], y: np.ndarray, *, window: int = 100
) -> dict:
    """Score current live artifact in fixed windows (its own cal+logit+topk)."""
    final = np.zeros(len(chunks), dtype=float)
    for start in range(0, len(chunks), window):
        sl = slice(start, start + window)
        scores = model.score_chunks(chunks[sl])
        final[sl] = np.asarray(scores, dtype=float)
    return eval_suite(y, final, f"live_model_bs{window}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--recent-val-days", type=int, default=4)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--large-ratio", type=float, default=1.25)
    ap.add_argument("--medium-ratio", type=float, default=0.40)
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="Required live-shaped reward gain over current to deploy.")
    ap.add_argument("--force-deploy", action="store_true",
                    help="Deploy candidate even if it does not beat current (NOT recommended).")
    args = ap.parse_args()

    if not args.examples.exists():
        raise SystemExit(f"Missing examples: {args.examples}")

    names = list(schema.FEATURE_NAMES)
    print(f"beat_v3_coherent features = {len(names)}", flush=True)
    print(f"Loading raw chunks from {args.examples}...", flush=True)
    chunks, y, dates = load_raw_examples(args.examples)
    unique = sorted(set(dates.tolist()))
    holdout_dates = unique[-args.holdout_days :]
    recent_dates = unique[-(args.holdout_days + args.recent_val_days) : -args.holdout_days]
    hold_set = set(holdout_dates)
    pool_mask = np.array([d not in hold_set for d in dates])
    hold_mask = ~pool_mask
    print(
        f"n={len(y)} dates={unique[0]}..{unique[-1]} holdout={holdout_dates} "
        f"lodo={recent_dates} pool={pool_mask.sum()} hold={hold_mask.sum()}",
        flush=True,
    )

    aug_cfg = LargeAugmentationConfig(
        large_ratio=float(args.large_ratio), medium_ratio=float(args.medium_ratio)
    )
    pool_idx = np.flatnonzero(pool_mask)
    hold_idx = np.flatnonzero(hold_mask)
    pool_chunks = [chunks[i] for i in pool_idx]
    pool_y = y[pool_mask]
    pool_dates = dates[pool_mask]

    print("Augmenting pool (large+medium)...", flush=True)
    train_chunks, train_y, train_dates, aug_stats = build_training_views(
        pool_chunks, pool_y, pool_dates.tolist(), aug_cfg, seed=args.seed
    )
    print(f"  aug stats={aug_stats}", flush=True)

    hold_chunks = [chunks[i] for i in hold_idx]
    hold_y = y[hold_mask]
    hold_dates = [str(d) for d in dates[hold_mask].tolist()]

    print("Featurizing train...", flush=True)
    X_train = featurize(train_chunks)
    y_train = np.asarray(train_y, dtype=np.int64)
    dates_train = np.asarray(train_dates)
    train_pool_mask = np.ones(len(y_train), dtype=bool)

    print("LODO + calibrate...", flush=True)
    oof, oy = lodo_oof(
        X_train, y_train, dates_train, train_pool_mask, recent_dates,
        X_hn=None, y_hn=None, hn_weight=0.0,
        half_life=args.half_life_days, seed=args.seed, model_factory=make_xgb,
    )
    cal, logit, _ = tune_cal_and_logit(oy, oof, args.seed)
    scored = apply_post(oof, calibrator=cal, logit=logit)
    m_lodo = eval_suite(oy, scored, "lodo")
    print(
        f"  LODO reward={m_lodo['reward']:.4f} ap={m_lodo['ap']:.4f} "
        f"bot@5fpr={m_lodo['bot_recall_at_5fpr']:.4f} logit={logit}",
        flush=True,
    )

    # Model fit on pool for holdout comparison.
    w_pool = recency_weights(dates_train, sorted(set(dates_train.tolist())), args.half_life_days)
    model_pool = make_xgb(args.seed)
    model_pool.fit(X_train, y_train, sample_weight=w_pool)

    # Build the SHARED live-shaped holdout (identical chunks + labels for both models).
    print("Building live-shaped holdout (80-100 hands)...", flush=True)
    live_chunks, live_y, _ = build_live_shaped_holdout(
        hold_chunks, hold_y, hold_dates, seed=args.seed + 7, ratio=1.0
    )
    print(f"  live-shaped n={len(live_y)}", flush=True)

    print("Candidate scoring on live-shaped holdout...", flush=True)
    X_live = featurize(live_chunks)
    cand_scores = apply_post(model_pool.predict_proba(X_live)[:, 1], calibrator=cal, logit=logit)
    cand_live = eval_suite(live_y, cand_scores, "candidate_live_shaped")
    cand_topk100 = simulate_batch_reward(live_y, cand_scores, batch_size=100, fraction=0.125)
    print(
        f"  CANDIDATE live reward={cand_live['reward']:.4f} "
        f"topk100={cand_topk100['reward']:.4f} ap={cand_live['ap']:.4f} "
        f"bot@5fpr={cand_live['bot_recall_at_5fpr']:.4f}",
        flush=True,
    )

    # Evaluate the CURRENT live artifact on the SAME chunks.
    current_path = args.out_dir / "current.joblib"
    current_metric = None
    current_version = None
    if current_path.exists():
        print("Scoring CURRENT live model on same holdout...", flush=True)
        live_model = CompetitiveMinerModel(model_path=current_path)
        current_version = live_model.model_version
        current_metric = score_live_model_windows(live_model, live_chunks, live_y, window=100)
        print(
            f"  CURRENT ({current_version}) live reward={current_metric['reward']:.4f} "
            f"ap={current_metric['ap']:.4f} bot@5fpr={current_metric['bot_recall_at_5fpr']:.4f}",
            flush=True,
        )
    else:
        print("  No current.joblib present; candidate will deploy as first model.", flush=True)

    cand_gate = cand_topk100["reward"]
    cur_gate = current_metric["reward"] if current_metric else -1.0
    gain = cand_gate - cur_gate
    beat = (current_metric is None) or (gain >= args.min_gain) or args.force_deploy

    print(
        f"\nGATE: candidate topk100={cand_gate:.4f} vs current={cur_gate:.4f} "
        f"gain={gain:+.4f} min_gain={args.min_gain:.4f} -> "
        f"{'DEPLOY' if beat else 'HOLD (keep current)'}",
        flush=True,
    )

    # Fit deploy model on ALL data (+aug) regardless; only WRITE if gate passes.
    print("Fitting deploy model on all+aug...", flush=True)
    deploy_chunks, deploy_y, deploy_dates, deploy_aug = build_training_views(
        chunks, y, dates.tolist(), aug_cfg, seed=args.seed + 99
    )
    X_deploy = featurize(deploy_chunks)
    y_deploy = np.asarray(deploy_y, dtype=np.int64)
    dates_deploy = np.asarray(deploy_dates)
    w_all = recency_weights(dates_deploy, sorted(set(dates_deploy.tolist())), args.half_life_days)
    model = make_xgb(args.seed)
    model.fit(X_deploy, y_deploy, sample_weight=w_all)

    trained_at = datetime.now(timezone.utc).isoformat()
    fingerprint = recipe_fingerprint(ROOT)
    report = {
        "trained_at_utc": trained_at,
        "model_name": "poker44-beat-v3-coherent",
        "model_version": "8.0.0",
        "n_features": len(names),
        "feature_set": "beat_v3_coherent",
        "recipe_fingerprint": fingerprint,
        "half_life_days": args.half_life_days,
        "score_remap": logit,
        "holdout_dates": holdout_dates,
        "recent_lodo_dates": recent_dates,
        "latest_source_date": unique[-1],
        "benchmark_n": int(len(y)),
        "augmentation": {"config": aug_cfg.as_dict(), "pool_stats": aug_stats, "deploy_stats": deploy_aug},
        "gate": {
            "candidate_live_topk100": cand_gate,
            "current_live_topk100": cur_gate,
            "current_version": current_version,
            "gain": gain,
            "min_gain": args.min_gain,
            "deployed": bool(beat),
            "forced": bool(args.force_deploy),
        },
        "metrics": {
            "lodo": m_lodo,
            "lodo_selection_score": selection_score(m_lodo),
            "candidate_live_shaped": cand_live,
            "candidate_live_shaped_topk100": cand_topk100,
            "current_live_shaped": current_metric,
        },
        "selection_policy": (
            "beat_v3 + coherent + large-chunk aug; deploy ONLY if candidate beats "
            "current live model on live-shaped topk100 by >= min_gain"
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
        "feature_set": "beat_v3_coherent",
        "holdout_dates": holdout_dates,
        "latest_source_date": unique[-1],
        "trained_at_utc": trained_at,
        "framework": "xgb_beat_v3_coherent",
        "recipe_fingerprint": fingerprint,
    }
    artifact = {
        "kind": "single",
        "models": [model],
        "weights": [1.0],
        "calibrator": cal,
        "feature_names": names,
        "feature_set": "beat_v3_coherent",
        "metadata": metadata,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if beat:
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
        atomic_joblib_dump(artifact, current_path)
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
                    "model_path": str(current_path),
                },
                indent=2,
            )
            + "\n",
        )
        print(f"DEPLOYED candidate -> {current_path}", flush=True)
    else:
        cand_path = args.out_dir / "candidate_beat_v3_coherent.joblib"
        atomic_joblib_dump(artifact, cand_path)
        atomic_write_text(
            args.out_dir / "candidate_report.json", json.dumps(report, indent=2) + "\n"
        )
        print(
            f"HELD: candidate did not beat current. Saved aside -> {cand_path}\n"
            f"Live model untouched.",
            flush=True,
        )

    print(json.dumps({"gate": report["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
