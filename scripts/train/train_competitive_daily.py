#!/usr/bin/env python3
"""Fast daily train: beat_v3 schema (competitive+FN+v3) + capacity XGB + LODO.

Beats xgb_v3_holdout on sealed 7/13-14 (reward). Used by daily_refresh_retrain.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import beat_v3_schema as schema
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
    """v3ish capacity config that beat xgb_v3_holdout on sealed 7/13-14."""
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


def load_labeled_competitive(path: Path):
    """Load beat_v3 features (competitive + FN patches + v3 cues)."""
    names = list(schema.FEATURE_NAMES)
    Xs, ys, dates = [], [], []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            ex = json.loads(line)
            hands = [prepare_hand_for_miner(h) for h in (ex.get("hands") or []) if isinstance(h, dict)]
            feat = schema.extract_chunk_features(hands)
            Xs.append(schema.features_to_vector(feat))
            ys.append(int(ex["label"]))
            dates.append(ex["sourceDate"])
            if (i + 1) % 500 == 0:
                print(f"  labeled {i+1}", flush=True)
    return (
        np.asarray(Xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(dates),
        names,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--recent-val-days", type=int, default=4)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--archive", action="store_true", help="Copy current.joblib to archive/ before overwrite")
    args = ap.parse_args()

    if not args.examples.exists():
        raise SystemExit(f"Missing examples: {args.examples}")

    names = list(schema.FEATURE_NAMES)
    print(f"Loading {args.examples} ({len(names)} features)...", flush=True)
    X, y, dates, names = load_labeled_competitive(args.examples)
    unique = sorted(set(dates.tolist()))
    holdout_dates = unique[-args.holdout_days :]
    recent_dates = unique[-(args.holdout_days + args.recent_val_days) : -args.holdout_days]
    hold_set = set(holdout_dates)
    pool = np.array([d not in hold_set for d in dates])
    hold = ~pool
    print(
        f"n={len(y)} dates={unique[0]}..{unique[-1]} "
        f"holdout={holdout_dates} lodo={recent_dates} pool={pool.sum()} hold={hold.sum()}",
        flush=True,
    )

    print("LODO + calibrate...", flush=True)
    oof, oy = lodo_oof(
        X,
        y,
        dates,
        pool,
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

    print("Sealed holdout...", flush=True)
    w_pool = recency_weights(dates[pool], sorted(set(dates[pool].tolist())), args.half_life_days)
    model_pool = make_xgb(args.seed)
    model_pool.fit(X[pool], y[pool], sample_weight=w_pool)
    ho = apply_post(model_pool.predict_proba(X[hold])[:, 1], calibrator=cal, logit=logit)
    m_ho = eval_suite(y[hold], ho, "holdout_sealed")
    m_topk = simulate_batch_reward(y[hold], ho)
    print(
        f"  SEALED reward={m_ho['reward']:.4f} ap={m_ho['ap']:.4f} "
        f"bot@5fpr={m_ho['bot_recall_at_5fpr']:.4f} fpr={m_ho['hard_fpr@0.5']:.4f}",
        flush=True,
    )

    print("Deploy fit on all labeled...", flush=True)
    w_all = recency_weights(dates, unique, args.half_life_days)
    model = make_xgb(args.seed)
    model.fit(X, y, sample_weight=w_all)

    trained_at = datetime.now(timezone.utc).isoformat()
    report = {
        "trained_at_utc": trained_at,
        "model_name": "poker44-beat-v3",
        "model_version": "7.0.0",
        "n_features": len(names),
        "feature_set": "beat_v3",
        "half_life_days": args.half_life_days,
        "score_remap": logit,
        "holdout_dates": holdout_dates,
        "recent_lodo_dates": recent_dates,
        "latest_source_date": unique[-1],
        "benchmark_n": int(len(y)),
        "label_counts": {
            "all": dict(Counter(y.tolist())),
            "holdout": dict(Counter(y[hold].tolist())),
        },
        "metrics": {
            "lodo": m_lodo,
            "lodo_selection_score": selection_score(m_lodo),
            "holdout_sealed": m_ho,
            "holdout_sealed_topk": m_topk,
        },
        "selection_policy": "daily recipe: beat_v3 (comp+FN+v3) capacity XGB + LODO; holdout sealed; deploy all",
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
        "framework": "xgb_beat_v3",
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
        print(f"Archived previous artifact under {arch}", flush=True)

    joblib.dump(artifact, out_path)
    (args.out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out_dir / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "batch_calibration": "topk_v1",
                "score_remap": logit,
                "model_name": report["model_name"],
                "model_version": report["model_version"],
                "latest_source_date": unique[-1],
                "model_path": str(out_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"latest_source_date": unique[-1], "metrics": report["metrics"]}, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
