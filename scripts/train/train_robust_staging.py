#!/usr/bin/env python3
"""Staging-only robust retrain: latest-in-train + strong reg + hard offline gate.

SAFETY
------
Default --out-dir is models/staging_robust/. This script NEVER writes
models/competitive/current.joblib. The live miner is undisturbed.

Method
------
1. Seal a random block of holdout days (NOT forced to be the latest dates).
2. Train on ALL remaining data, including the most recent dates.
3. Stronger XGBoost regularization (shallower, higher lambda/min_child).
4. Estimate vs live 8.0.0 with:
   - live-shaped chunks (80-100 hands)
   - windowed reward (batch size 100, real serving path)
   - low bot-rate stress (downsample positives inside windows)
5. Write candidate + estimation_report.json only. Deploy is a separate manual step.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker44.artifact_io import atomic_joblib_dump, atomic_write_text, recipe_fingerprint
from poker44.batch_calibration import apply_batch_safety_topk_v1
from poker44.large_chunk_augment import (
    LargeAugmentationConfig,
    build_live_shaped_holdout,
    build_training_views,
)
from poker44.miner_inference import CompetitiveMinerModel
from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner
from scripts.train.train_competitive_v3 import (
    apply_post,
    eval_suite,
    recency_weights,
    tune_cal_and_logit,
)

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT = ROOT / "models" / "staging_robust"
LIVE_MODEL = ROOT / "models" / "competitive" / "current.joblib"
LIVE_DIR_FORBIDDEN = (ROOT / "models" / "competitive").resolve()

MODEL_VERSION = "8.1.0-robust"


def make_xgb_strong(seed: int) -> XGBClassifier:
    """Anti-overfit XGB: shallower trees, heavier L2, stronger leaf constraints."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.75,
        colsample_bytree=0.55,
        min_child_weight=8,
        reg_lambda=12.0,
        reg_alpha=1.0,
        gamma=0.5,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )


def load_schema(feature_set: str):
    if feature_set == "beat_v3":
        from features import beat_v3_schema as schema
    elif feature_set == "beat_v3_coherent":
        from features import beat_v3_coherent_schema as schema
    else:
        raise SystemExit(f"Unknown feature_set: {feature_set}")
    return schema


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


def featurize(schema, chunks: list[list[dict]]) -> np.ndarray:
    names = list(schema.FEATURE_NAMES)
    rows = []
    for i, hands in enumerate(chunks):
        feat = schema.extract_chunk_features(hands)
        rows.append([float(feat.get(n, 0.0)) for n in names])
        if (i + 1) % 500 == 0:
            print(f"  featurized {i + 1}/{len(chunks)}", flush=True)
    return np.asarray(rows, dtype=np.float64)


def pick_random_holdout_dates(
    unique: list[str],
    dates: np.ndarray,
    n_days: int,
    seed: int,
    *,
    min_per_day: int = 80,
    keep_latest_in_train: int = 2,
) -> list[str]:
    """Seal random denser days; always keep the most recent dates in training.

    Avoids sparse early calendar days (too few chunks for a meaningful estimate)
    and never holds out the last ``keep_latest_in_train`` dates.
    """
    if n_days <= 0:
        raise SystemExit(f"holdout-days={n_days} invalid")
    counts = {d: int((dates == d).sum()) for d in unique}
    cutoff = unique[:-keep_latest_in_train] if keep_latest_in_train > 0 else list(unique)
    eligible = [d for d in cutoff if counts.get(d, 0) >= min_per_day]
    if len(eligible) < n_days:
        eligible = sorted(cutoff, key=lambda d: counts.get(d, 0), reverse=True)
    if len(eligible) < n_days:
        raise SystemExit(
            f"Not enough eligible holdout days (need {n_days}, have {len(eligible)})."
        )
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=n_days, replace=False).tolist())
    return chosen


def score_serving_windows(model: CompetitiveMinerModel, chunks, y, *, window: int = 100) -> dict:
    y = np.asarray(y)
    final = np.zeros(len(y), dtype=float)
    for start in range(0, len(y), window):
        sl = slice(start, start + window)
        final[sl] = np.asarray(model.score_chunks(chunks[sl]), dtype=float)
    pooled = eval_suite(y, final, f"serving_pooled_bs{window}")
    window_rewards = []
    zero_gates = 0
    for start in range(0, len(y), window):
        sl = slice(start, min(start + window, len(y)))
        if sl.stop - sl.start < 10:
            continue
        rew, det = reward(final[sl], y[sl])
        window_rewards.append(float(rew))
        if float(det.get("threshold_sanity_quality", 0.0)) <= 0:
            zero_gates += 1
    return {
        "pooled": pooled,
        "windowed_mean_reward": float(np.mean(window_rewards)) if window_rewards else 0.0,
        "windowed_std_reward": float(np.std(window_rewards)) if window_rewards else 0.0,
        "n_windows": len(window_rewards),
        "zero_gated_windows": zero_gates,
        "scores": final,
    }


def low_bot_rate_stress(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    target_bot_rate: float,
    window: int = 100,
    seed: int = 0,
    topk_fraction: float = 0.125,
) -> dict:
    """Apply top-K inside windows after downsampling bots to target rate; mean reward."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    rews = []
    zero_gates = 0
    for start in range(0, len(y), window):
        sl = slice(start, min(start + window, len(y)))
        yy = y[sl]
        ss = scores[sl]
        pos = np.flatnonzero(yy == 1)
        neg = np.flatnonzero(yy == 0)
        if len(neg) == 0 or len(pos) == 0:
            continue
        keep_pos = max(1, int(round(target_bot_rate * len(neg) / max(1e-9, 1.0 - target_bot_rate))))
        if keep_pos < len(pos):
            kp = rng.choice(pos, keep_pos, replace=False)
            idx = np.concatenate([neg, kp])
        else:
            idx = np.arange(len(yy))
        # Use rank order of raw-ish scores then top-K (approximate serving budget)
        ranked = apply_batch_safety_topk_v1(ss[idx], max_positive_fraction=topk_fraction)
        rew, det = reward(np.asarray(ranked, dtype=float), yy[idx])
        rews.append(float(rew))
        if float(det.get("threshold_sanity_quality", 0.0)) <= 0:
            zero_gates += 1
    return {
        "target_bot_rate": target_bot_rate,
        "mean_reward": float(np.mean(rews)) if rews else 0.0,
        "std_reward": float(np.std(rews)) if rews else 0.0,
        "n_windows": len(rews),
        "zero_gated_windows": zero_gates,
    }


def assert_staging_only(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    if resolved == LIVE_DIR_FORBIDDEN or LIVE_DIR_FORBIDDEN in resolved.parents:
        # Allow only if explicitly under a non-competitive path; competitive is forbidden.
        if resolved == LIVE_DIR_FORBIDDEN or resolved.parent == LIVE_DIR_FORBIDDEN:
            raise SystemExit(
                f"REFUSING to write under live dir {LIVE_DIR_FORBIDDEN}. "
                "Use --out-dir models/staging_robust (default)."
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--live-model", type=Path, default=LIVE_MODEL,
                    help="Read-only path to live artifact for comparison.")
    ap.add_argument("--feature-set", choices=("beat_v3", "beat_v3_coherent"), default="beat_v3")
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--large-ratio", type=float, default=1.25)
    ap.add_argument("--medium-ratio", type=float, default=0.40)
    ap.add_argument("--allow-live-write", action="store_true",
                    help="DANGEROUS: allow out-dir under models/competitive. Default refused.")
    args = ap.parse_args()

    if not args.allow_live_write:
        assert_staging_only(args.out_dir)

    if not args.examples.exists():
        raise SystemExit(f"Missing examples: {args.examples}")

    schema = load_schema(args.feature_set)
    names = list(schema.FEATURE_NAMES)
    print(
        f"feature_set={args.feature_set} n_features={len(names)} "
        f"out_dir={args.out_dir} (staging-only)",
        flush=True,
    )

    print(f"Loading {args.examples}...", flush=True)
    chunks, y, dates = load_raw_examples(args.examples)
    unique = sorted(set(dates.tolist()))
    holdout_dates = pick_random_holdout_dates(
        unique, dates, args.holdout_days, args.seed, min_per_day=80, keep_latest_in_train=2
    )
    hold_set = set(holdout_dates)
    pool_mask = np.array([d not in hold_set for d in dates])
    hold_mask = ~pool_mask
    latest = unique[-1]
    print(
        f"n={len(y)} dates={unique[0]}..{latest} "
        f"RANDOM holdout={holdout_dates} "
        f"(latest {latest} {'IN TRAIN' if latest not in hold_set else 'IN HOLDOUT'}) "
        f"pool={pool_mask.sum()} hold={hold_mask.sum()}",
        flush=True,
    )

    aug_cfg = LargeAugmentationConfig(
        large_ratio=float(args.large_ratio), medium_ratio=float(args.medium_ratio)
    )
    pool_chunks = [chunks[i] for i in np.flatnonzero(pool_mask)]
    pool_y = y[pool_mask]
    pool_dates = dates[pool_mask]

    print("Augmenting train pool (includes latest if not in holdout)...", flush=True)
    train_chunks, train_y, train_dates, aug_stats = build_training_views(
        pool_chunks, pool_y, pool_dates.tolist(), aug_cfg, seed=args.seed
    )
    train_y = np.asarray(train_y, dtype=np.int64)
    train_dates = np.asarray([str(d) for d in train_dates])
    print(f"  aug={aug_stats} n_train={len(train_y)}", flush=True)

    print("Featurizing train...", flush=True)
    X_train = featurize(schema, train_chunks)
    w = recency_weights(train_dates, sorted(set(train_dates.tolist())), args.half_life_days)

    # Internal val slice for cal/logit: last 15% of shuffled train by date strata approx
    rng = np.random.default_rng(args.seed + 1)
    order = rng.permutation(len(train_y))
    val_n = max(50, int(0.15 * len(train_y)))
    val_idx, fit_idx = order[:val_n], order[val_n:]

    print("Fitting strong-reg XGB...", flush=True)
    model = make_xgb_strong(args.seed)
    model.fit(X_train[fit_idx], train_y[fit_idx], sample_weight=w[fit_idx])

    raw_val = model.predict_proba(X_train[val_idx])[:, 1]
    cal, logit, _ = tune_cal_and_logit(train_y[val_idx], raw_val, args.seed)
    print(f"  cal/logit tuned: logit={logit}", flush=True)

    # Refit on full train pool for candidate artifact
    model_full = make_xgb_strong(args.seed)
    model_full.fit(X_train, train_y, sample_weight=w)

    trained_at = datetime.now(timezone.utc).isoformat()
    fingerprint = recipe_fingerprint(ROOT)
    tag = f"{args.feature_set}_strongreg"
    metadata = {
        "model_name": f"poker44-robust-{args.feature_set}",
        "model_version": MODEL_VERSION,
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
        "feature_set": args.feature_set,
        "holdout_dates": holdout_dates,
        "latest_source_date": latest,
        "trained_at_utc": trained_at,
        "framework": "xgb_strongreg_staging",
        "recipe_fingerprint": fingerprint,
        "xgb_profile": "strong_reg_depth3_lambda12",
    }
    artifact = {
        "kind": "single",
        "models": [model_full],
        "weights": [1.0],
        "calibrator": cal,
        "feature_names": names,
        "feature_set": args.feature_set,
        "metadata": metadata,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = args.out_dir / f"candidate_{tag}.joblib"
    atomic_joblib_dump(artifact, cand_path)
    print(f"Wrote staging candidate (live untouched): {cand_path}", flush=True)

    # --- estimation holdout (sealed random days), live-shaped ---
    hold_chunks = [chunks[i] for i in np.flatnonzero(hold_mask)]
    hold_y = y[hold_mask]
    hold_dates = [str(d) for d in dates[hold_mask].tolist()]
    print("Building live-shaped estimation holdout...", flush=True)
    live_chunks, live_y, _ = build_live_shaped_holdout(
        hold_chunks, hold_y, hold_dates, seed=args.seed + 7, ratio=1.0
    )
    live_y = np.asarray(live_y)
    print(f"  live-shaped n={len(live_y)} bot_rate={live_y.mean():.2f}", flush=True)

    print("Scoring CANDIDATE via serving path...", flush=True)
    cand_model = CompetitiveMinerModel(model_path=cand_path)
    cand_eval = score_serving_windows(cand_model, live_chunks, live_y, window=100)
    # For low-bot stress use pre-topk calibrated path approx: re-score raw then stress
    # Serving scores already have top-K; for stress use model internals: raw+cal+logit then stress topk
    from poker44.miner_inference import CompetitiveMinerModel as CMM

    # Get remapped scores without top-K by temporarily forcing batch mode
    import os

    os.environ["POKER44_BATCH_CALIBRATION"] = "none"
    cand_raw_model = CMM(model_path=cand_path)
    cand_raw = np.zeros(len(live_y))
    for s in range(0, len(live_y), 100):
        sl = slice(s, s + 100)
        cand_raw[sl] = np.asarray(cand_raw_model.score_chunks(live_chunks[sl]), dtype=float)
    os.environ.pop("POKER44_BATCH_CALIBRATION", None)

    cand_stress = {
        f"bot_{int(r * 100)}": low_bot_rate_stress(
            cand_raw, live_y, target_bot_rate=r, seed=args.seed + 11
        )
        for r in (0.20, 0.10, 0.05)
    }

    live_eval = None
    live_stress = None
    live_version = None
    if args.live_model.exists():
        print(f"Scoring LIVE (read-only) {args.live_model}...", flush=True)
        live_model = CompetitiveMinerModel(model_path=args.live_model)
        live_version = live_model.model_version
        live_eval = score_serving_windows(live_model, live_chunks, live_y, window=100)
        os.environ["POKER44_BATCH_CALIBRATION"] = "none"
        live_raw_model = CMM(model_path=args.live_model)
        live_raw = np.zeros(len(live_y))
        for s in range(0, len(live_y), 100):
            sl = slice(s, s + 100)
            live_raw[sl] = np.asarray(live_raw_model.score_chunks(live_chunks[sl]), dtype=float)
        os.environ.pop("POKER44_BATCH_CALIBRATION", None)
        live_stress = {
            f"bot_{int(r * 100)}": low_bot_rate_stress(
                live_raw, live_y, target_bot_rate=r, seed=args.seed + 11
            )
            for r in (0.20, 0.10, 0.05)
        }
    else:
        print("No live model found for comparison.", flush=True)

    report = {
        "trained_at_utc": trained_at,
        "staging_only": True,
        "deployed_to_live": False,
        "model_version": MODEL_VERSION,
        "feature_set": args.feature_set,
        "n_features": len(names),
        "holdout_dates": holdout_dates,
        "latest_source_date": latest,
        "latest_in_training": latest not in hold_set,
        "augmentation": {"config": aug_cfg.as_dict(), "stats": aug_stats},
        "score_remap": logit,
        "candidate_path": str(cand_path),
        "live_version": live_version,
        "estimation": {
            "candidate": {
                "windowed_mean_reward": cand_eval["windowed_mean_reward"],
                "pooled_reward": cand_eval["pooled"]["reward"],
                "pooled_ap": cand_eval["pooled"]["ap"],
                "pooled_bot_recall_at_5fpr": cand_eval["pooled"]["bot_recall_at_5fpr"],
                "zero_gated_windows": cand_eval["zero_gated_windows"],
                "low_bot_stress": cand_stress,
            },
            "live": None
            if live_eval is None
            else {
                "windowed_mean_reward": live_eval["windowed_mean_reward"],
                "pooled_reward": live_eval["pooled"]["reward"],
                "pooled_ap": live_eval["pooled"]["ap"],
                "pooled_bot_recall_at_5fpr": live_eval["pooled"]["bot_recall_at_5fpr"],
                "zero_gated_windows": live_eval["zero_gated_windows"],
                "low_bot_stress": live_stress,
            },
            "deltas_windowed": None
            if live_eval is None
            else {
                "windowed_mean_reward": cand_eval["windowed_mean_reward"]
                - live_eval["windowed_mean_reward"],
                "pooled_reward": cand_eval["pooled"]["reward"] - live_eval["pooled"]["reward"],
            },
        },
        "selection_policy": (
            "staging robust: latest-in-train when not in random holdout; strong-reg XGB; "
            "estimate via windowed + low-bot stress; NEVER auto-deploy to live"
        ),
    }
    report_path = args.out_dir / f"estimation_{tag}.json"
    atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")

    print("\n======== ESTIMATION (live miner NOT changed) ========", flush=True)
    print(json.dumps(report["estimation"], indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)
    print("Deploy only after you review and explicitly approve.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
