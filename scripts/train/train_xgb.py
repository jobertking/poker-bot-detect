#!/usr/bin/env python3
"""Train an XGBoost bot-risk model on Poker44 chunk features."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker44.score.scoring import reward

DEFAULT_FEATURES = ROOT / "data" / "benchmark" / "features" / "features.npz"
DEFAULT_OUT = ROOT / "models" / "xgb_v1"


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rew, detail = reward(y_score, y_true)
    out = {
        "n": int(y_true.size),
        "pos": int(np.sum(y_true == 1)),
        "neg": int(np.sum(y_true == 0)),
        "ap": float(average_precision_score(y_true, y_score)) if np.any(y_true == 1) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score))
        if np.unique(y_true).size > 1
        else float("nan"),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_score)),
        "reward": float(rew),
        "bot_recall_at_5fpr": float(detail["bot_recall"]),
        "fpr_at_best_recall": float(detail["fpr"]),
        "hard_bot_recall@0.5": float(detail["hard_bot_recall"]),
        "hard_fpr@0.5": float(detail["hard_fpr"]),
        "threshold_sanity_quality": float(detail["threshold_sanity_quality"]),
        "mean_score": float(np.mean(y_score)),
        "mean_score_bot": float(np.mean(y_score[y_true == 1])) if np.any(y_true == 1) else None,
        "mean_score_human": float(np.mean(y_score[y_true == 0])) if np.any(y_true == 0) else None,
    }
    return out


def _date_masks(
    source_dates: np.ndarray,
    splits: np.ndarray,
    *,
    holdout_days: int,
) -> dict[str, np.ndarray]:
    unique_dates = sorted(set(source_dates.tolist()))
    if holdout_days <= 0 or holdout_days >= len(unique_dates):
        raise ValueError(f"holdout_days={holdout_days} invalid for {len(unique_dates)} dates")

    holdout_dates = set(unique_dates[-holdout_days:])
    train_pool_dates = set(unique_dates[:-holdout_days])

    is_holdout = np.array([d in holdout_dates for d in source_dates], dtype=bool)
    in_pool = np.array([d in train_pool_dates for d in source_dates], dtype=bool)

    # Prefer official split labels inside the train pool; keep leftovers in train.
    split_train = np.array([s == "train" for s in splits], dtype=bool)
    split_val = np.array([s == "validation" for s in splits], dtype=bool)

    train_mask = in_pool & split_train
    val_mask = in_pool & split_val

    # If val is too small, move the last non-holdout date fully into val.
    if int(val_mask.sum()) < 50:
        val_date = unique_dates[-(holdout_days + 1)]
        val_mask = in_pool & (source_dates == val_date)
        train_mask = in_pool & ~val_mask

    # Any pool rows not covered (shouldn't happen) go to train.
    uncovered = in_pool & ~train_mask & ~val_mask
    train_mask |= uncovered

    return {
        "train": train_mask,
        "val": val_mask,
        "holdout": is_holdout,
        "holdout_dates": sorted(holdout_dates),
        "train_dates": sorted(set(source_dates[train_mask].tolist())),
        "val_dates": sorted(set(source_dates[val_mask].tolist())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--holdout-days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int64)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    source_dates = np.asarray(data["sourceDates"])
    splits = np.asarray(data["splits"])
    example_ids = np.asarray(data["example_ids"])

    masks = _date_masks(source_dates, splits, holdout_days=args.holdout_days)
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_val, y_val = X[masks["val"]], y[masks["val"]]
    X_hold, y_hold = X[masks["holdout"]], y[masks["holdout"]]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=4,
        early_stopping_rounds=40,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_hold = model.predict_proba(X_hold)[:, 1]

    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-xgb-v1",
        "model_version": "1.0.0",
        "features_path": str(args.features),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
        "splits": {
            "train_n": int(masks["train"].sum()),
            "val_n": int(masks["val"].sum()),
            "holdout_n": int(masks["holdout"].sum()),
            "train_label_counts": dict(Counter(y_train.tolist())),
            "val_label_counts": dict(Counter(y_val.tolist())),
            "holdout_label_counts": dict(Counter(y_hold.tolist())),
            "holdout_dates": masks["holdout_dates"],
            "val_dates": masks["val_dates"],
            "train_dates_n": len(masks["train_dates"]),
        },
        "metrics": {
            "train": _metrics(y_train, proba_train),
            "val": _metrics(y_val, proba_val),
            "holdout": _metrics(y_hold, proba_hold),
        },
        "params": model.get_params(),
    }

    # Top features by importance
    importances = model.feature_importances_
    top_idx = np.argsort(-importances)[:20]
    report["top_features"] = [
        {"name": feature_names[i], "importance": float(importances[i])} for i in top_idx
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.json"
    model.save_model(model_path)

    # Also keep a numpy-friendly bundle for inference without re-reading names.
    bundle_path = args.out_dir / "bundle.npz"
    np.savez_compressed(
        bundle_path,
        feature_names=np.asarray(feature_names),
        holdout_dates=np.asarray(masks["holdout_dates"]),
        holdout_example_ids=example_ids[masks["holdout"]],
        holdout_y=y_hold,
        holdout_proba=proba_hold,
    )

    report_path = args.out_dir / "train_report.json"
    # params may contain non-JSON values
    safe_params = {}
    for k, v in report["params"].items():
        try:
            json.dumps(v)
            safe_params[k] = v
        except TypeError:
            safe_params[k] = str(v)
    report["params"] = safe_params
    report["paths"] = {
        "model_json": str(model_path),
        "bundle_npz": str(bundle_path),
        "report": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "model": report["model_name"],
            "best_iteration": report["best_iteration"],
            "splits": report["splits"],
            "metrics": report["metrics"],
            "top_features": report["top_features"][:10],
            "paths": report["paths"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
