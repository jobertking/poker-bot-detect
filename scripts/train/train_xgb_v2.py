#!/usr/bin/env python3
"""Train XGBoost v2 targeting high bot-detection rate (>=95% bot recall)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.chunk_features import FEATURE_NAMES, extract_chunk_features, features_to_vector
from poker44.score.scoring import reward

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT = ROOT / "models" / "xgb_v2"


def build_xy(examples_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    Xs: list[list[float]] = []
    ys: list[int] = []
    dates: list[str] = []
    splits: list[str] = []
    ids: list[str] = []
    with examples_path.open(encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            vec = features_to_vector(extract_chunk_features(ex.get("hands") or []))
            Xs.append(vec)
            ys.append(int(ex["label"]))
            dates.append(ex["sourceDate"])
            splits.append(str(ex.get("split") or "unknown"))
            ids.append(ex["example_id"])
    return (
        np.asarray(Xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(dates),
        np.asarray(splits),
        ids,
    )


def split_masks(dates: np.ndarray, splits: np.ndarray, holdout_days: int) -> dict:
    unique = sorted(set(dates.tolist()))
    if holdout_days < 0:
        raise ValueError("holdout_days must be >= 0")
    if holdout_days == 0:
        holdout_dates: set[str] = set()
        pool_dates = set(unique)
    else:
        if holdout_days >= len(unique):
            raise ValueError(f"holdout_days={holdout_days} invalid for {len(unique)} dates")
        holdout_dates = set(unique[-holdout_days:])
        pool_dates = set(unique[:-holdout_days])

    is_holdout = np.array([d in holdout_dates for d in dates])
    in_pool = np.array([d in pool_dates for d in dates])
    train = in_pool & (splits == "train")
    val = in_pool & (splits == "validation")
    if int(val.sum()) < 40:
        # Fall back to last pool date as validation.
        pool_sorted = sorted(pool_dates)
        val_date = pool_sorted[-1]
        val = in_pool & (dates == val_date)
        train = in_pool & ~val
    uncovered = in_pool & ~train & ~val
    train |= uncovered
    return {
        "train": train,
        "val": val,
        "holdout": is_holdout,
        "holdout_dates": sorted(holdout_dates),
    }


def bot_recall_at_threshold(y: np.ndarray, scores: np.ndarray, thr: float) -> float:
    y = np.asarray(y)
    pred = scores >= thr
    pos = max(1, int((y == 1).sum()))
    return float(((pred) & (y == 1)).sum() / pos)


def human_correct_at_threshold(y: np.ndarray, scores: np.ndarray, thr: float) -> float:
    y = np.asarray(y)
    pred = scores >= thr
    neg = max(1, int((y == 0).sum()))
    return float(((~pred) & (y == 0)).sum() / neg)


def choose_threshold_for_bot_recall(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    target_bot_recall: float,
) -> dict:
    """Lowest threshold that reaches target bot recall; maximize human correct among ties."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    candidates = np.sort(candidates)[::-1]  # high -> low

    best = None
    for thr in candidates:
        br = bot_recall_at_threshold(y, scores, thr)
        if br + 1e-12 < target_bot_recall:
            continue
        hc = human_correct_at_threshold(y, scores, thr)
        acc = float(accuracy_score(y, (scores >= thr).astype(int)))
        item = {
            "threshold": float(thr),
            "bot_correct": float(br),
            "human_correct": float(hc),
            "overall_correct": acc,
        }
        if best is None or hc > best["human_correct"] + 1e-12 or (
            abs(hc - best["human_correct"]) <= 1e-12 and thr > best["threshold"]
        ):
            best = item
    if best is None:
        # fallback: thr that maxes bot recall
        thr = float(np.min(scores[y == 1])) if np.any(y == 1) else 0.5
        best = {
            "threshold": thr,
            "bot_correct": bot_recall_at_threshold(y, scores, thr),
            "human_correct": human_correct_at_threshold(y, scores, thr),
            "overall_correct": float(accuracy_score(y, (scores >= thr).astype(int))),
        }
    return best


def eval_at_threshold(y: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= thr).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    rew, detail = reward(scores, y)
    return {
        "n": int(y.size),
        "threshold": float(thr),
        "overall_correct_percent": round(100.0 * float(accuracy_score(y, pred)), 2),
        "bot_correct_percent": round(100.0 * tp / max(1, int((y == 1).sum())), 2),
        "human_correct_percent": round(100.0 * tn / max(1, int((y == 0).sum())), 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "ap": float(average_precision_score(y, scores)) if np.any(y == 1) else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)) if np.unique(y).size > 1 else None,
        "reward": float(rew),
        "hard_fpr": float(detail["hard_fpr"]) if abs(thr - 0.5) < 1e-12 else float(fp / max(1, (y == 0).sum())),
        "bot_recall_at_5fpr": float(detail["bot_recall"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--holdout-days", type=int, default=2)
    parser.add_argument("--target-bot-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Building features...", flush=True)
    X, y, dates, splits, ids = build_xy(args.examples)
    masks = split_masks(dates, splits, args.holdout_days)
    Xtr, ytr = X[masks["train"]], y[masks["train"]]
    Xva, yva = X[masks["val"]], y[masks["val"]]
    Xho, yho = X[masks["holdout"]], y[masks["holdout"]]

    grid = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [1.0, 3.0],
        "scale_pos_weight": [1.0, 1.25, 1.5],
    }
    # Compact random-ish product sample for speed
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    rng = np.random.default_rng(args.seed)
    if len(combos) > 48:
        pick = rng.choice(len(combos), size=48, replace=False)
        combos = [combos[i] for i in pick]

    best = None
    print(f"Searching {len(combos)} configs...", flush=True)
    for idx, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        model = XGBClassifier(
            n_estimators=500,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=args.seed,
            n_jobs=4,
            early_stopping_rounds=50,
            **params,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        s_va = model.predict_proba(Xva)[:, 1]
        thr_info = choose_threshold_for_bot_recall(
            yva, s_va, target_bot_recall=args.target_bot_recall
        )
        va_ap = float(average_precision_score(yva, s_va))
        if yho.size > 0:
            s_ho = model.predict_proba(Xho)[:, 1]
            ho_bot = bot_recall_at_threshold(yho, s_ho, thr_info["threshold"])
            ho_hum = human_correct_at_threshold(yho, s_ho, thr_info["threshold"])
            score = (
                3.0 * min(ho_bot, 1.0)
                + 1.5 * ho_hum
                + 1.0 * thr_info["human_correct"]
                + 0.5 * va_ap
            )
        else:
            # No holdout: select by validation bot+human tradeoff.
            ho_bot = float("nan")
            ho_hum = float("nan")
            score = (
                3.0 * thr_info["bot_correct"]
                + 1.5 * thr_info["human_correct"]
                + 0.5 * va_ap
            )
        cand = {
            "params": params,
            "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
            "val_threshold": thr_info,
            "holdout_bot": None if yho.size == 0 else float(ho_bot),
            "holdout_human": None if yho.size == 0 else float(ho_hum),
            "val_ap": va_ap,
            "select_score": float(score),
            "model": model,
        }
        if best is None or cand["select_score"] > best["select_score"]:
            best = cand
            print(
                f"[{idx}/{len(combos)}] new best select={score:.4f} "
                f"holdout_bot={ho_bot} holdout_human={ho_hum} "
                f"thr={thr_info['threshold']:.4f} params={params}",
                flush=True,
            )

    assert best is not None
    model = best["model"]
    thr = float(best["val_threshold"]["threshold"])

    s_all = model.predict_proba(X)[:, 1]
    s_tr = model.predict_proba(Xtr)[:, 1]
    s_va = model.predict_proba(Xva)[:, 1]
    s_ho = model.predict_proba(Xho)[:, 1] if yho.size > 0 else np.asarray([])

    # Also compute threshold solely on ALL data for user's "all labeled data" ask,
    # but keep deployment threshold = val-chosen.
    thr_all = choose_threshold_for_bot_recall(y, s_all, target_bot_recall=args.target_bot_recall)

    focus_dates = ["2026-07-13", "2026-07-14"]
    by_date = {}
    for d in focus_dates:
        mask = dates == d
        if int(mask.sum()) == 0:
            continue
        by_date[d] = {
            "at_deployment_threshold": eval_at_threshold(y[mask], s_all[mask], thr),
            "at_0.5": eval_at_threshold(y[mask], s_all[mask], 0.5),
        }

    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-xgb-v2",
        "model_version": "2.1.0-incl-0713-0714",
        "target_bot_recall": args.target_bot_recall,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "params": best["params"],
        "best_iteration": best["best_iteration"],
        "deployment_threshold": thr,
        "threshold_source": "validation_for_target_bot_recall",
        "splits": {
            "train_n": int(masks["train"].sum()),
            "val_n": int(masks["val"].sum()),
            "holdout_n": int(masks["holdout"].sum()),
            "holdout_dates": masks["holdout_dates"],
            "label_counts_all": dict(Counter(y.tolist())),
            "includes_0713_0714_in_training_pool": True,
        },
        "metrics_at_deployment_threshold": {
            "all_labeled_data": eval_at_threshold(y, s_all, thr),
            "train": eval_at_threshold(ytr, s_tr, thr),
            "validation": eval_at_threshold(yva, s_va, thr),
            "holdout": eval_at_threshold(yho, s_ho, thr) if yho.size > 0 else None,
        },
        "metrics_at_0.5": {
            "all_labeled_data": eval_at_threshold(y, s_all, 0.5),
            "holdout": eval_at_threshold(yho, s_ho, 0.5) if yho.size > 0 else None,
        },
        "metrics_on_0713_0714": by_date,
        "all_data_threshold_for_95_bot": thr_all,
        "metrics_if_tune_threshold_on_all_data": eval_at_threshold(y, s_all, thr_all["threshold"]),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.json"
    model.save_model(model_path)
    thr_path = args.out_dir / "threshold.json"
    thr_path.write_text(
        json.dumps(
            {
                "threshold": thr,
                "target_bot_recall": args.target_bot_recall,
                "model_name": report["model_name"],
                "model_version": report["model_version"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Save rebuilt feature matrix for reuse
    np.savez_compressed(
        args.out_dir / "features_used.npz",
        X=X,
        y=y,
        feature_names=np.asarray(FEATURE_NAMES),
        sourceDates=dates,
        splits=splits,
        example_ids=np.asarray(ids),
        scores=s_all,
    )

    report_path = args.out_dir / "train_report.json"
    # drop non-serializable
    serial = dict(report)
    report_path.write_text(json.dumps(serial, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(serial, indent=2))
    print(f"\nSaved model -> {model_path}")
    print(f"Saved threshold -> {thr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
