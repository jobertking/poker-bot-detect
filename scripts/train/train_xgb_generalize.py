#!/usr/bin/env python3
"""XGBoost-only: recent LODO validation focus, lower FPR, calibrate + tune threshold."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.chunk_features import FEATURE_NAMES, extract_chunk_features, features_to_vector
from poker44.score.scoring import reward

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT = ROOT / "models" / "xgb_v2_gen"


def pct(x: float) -> float:
    return round(100.0 * float(x), 2)


def build_xy(path: Path):
    Xs, ys, dates = [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            Xs.append(features_to_vector(extract_chunk_features(ex.get("hands") or [])))
            ys.append(int(ex["label"]))
            dates.append(ex["sourceDate"])
    return np.asarray(Xs, np.float64), np.asarray(ys, np.int64), np.asarray(dates)


def recency_weights(dates: np.ndarray, unique_sorted: list[str], half_life_days: float) -> np.ndarray:
    idx = {d: i for i, d in enumerate(unique_sorted)}
    max_i = max(len(unique_sorted) - 1, 1)
    ages = np.array([max_i - idx[d] for d in dates], dtype=np.float64)
    decay = math.log(2.0) / max(half_life_days, 1e-6)
    w = np.exp(-decay * ages)
    return w / max(w.mean(), 1e-12)


def eval_at(y, scores, thr: float) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y.size == 0:
        return {"n": 0}
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    rew, detail = reward(scores, y)
    return {
        "n": int(y.size),
        "threshold": float(thr),
        "overall_correct_percent": pct(accuracy_score(y, pred)),
        "bot_correct_percent": pct(tp / max(1, int((y == 1).sum()))),
        "human_correct_percent": pct(tn / max(1, int((y == 0).sum()))),
        "precision_bot_percent": pct(precision_score(y, pred, zero_division=0)),
        "recall_bot_percent": pct(recall_score(y, pred, zero_division=0)),
        "f1_bot_percent": pct(f1_score(y, pred, zero_division=0)),
        "fpr_percent": pct(fp / max(1, int((y == 0).sum()))),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "ap": float(average_precision_score(y, scores)) if np.any(y == 1) else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)) if np.unique(y).size > 1 else None,
        "log_loss": float(log_loss(y, np.clip(scores, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(y, scores)),
        "subnet_reward": float(rew),
        "bot_recall_at_5fpr": float(detail["bot_recall"]),
    }


def choose_threshold(y, scores, *, min_bot: float, max_fpr: float) -> dict:
    """Prefer joint: overall>=95, bot>=min_bot, fpr<=max_fpr. Else best tradeoff."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rows = []
    for thr in np.sort(np.unique(np.concatenate([scores, [0.0, 0.5, 1.0]]))):
        pred = scores >= thr
        pos = max(1, int((y == 1).sum()))
        neg = max(1, int((y == 0).sum()))
        bot = float(((pred) & (y == 1)).sum() / pos)
        fpr = float(((pred) & (y == 0)).sum() / neg)
        overall = float((pred.astype(int) == y).mean())
        rows.append(
            {
                "threshold": float(thr),
                "bot_correct": bot,
                "fpr": fpr,
                "human_correct": 1.0 - fpr,
                "overall_correct": overall,
            }
        )

    perfect = [
        r
        for r in rows
        if r["overall_correct"] >= 0.95 and r["bot_correct"] >= min_bot and r["fpr"] <= max_fpr
    ]
    if perfect:
        perfect.sort(key=lambda r: (-r["fpr"], r["overall_correct"], r["bot_correct"]), reverse=True)
        best = perfect[0]
        best["mode"] = "hit_all_goals"
        return best

    bot_fpr = [r for r in rows if r["bot_correct"] >= min_bot and r["fpr"] <= max_fpr]
    if bot_fpr:
        bot_fpr.sort(key=lambda r: (r["overall_correct"], -r["fpr"]), reverse=True)
        best = bot_fpr[0]
        best["mode"] = "hit_bot_and_fpr"
        return best

    # Soft: maximize overall under FPR cap, then maximize bot under overall high.
    under_fpr = [r for r in rows if r["fpr"] <= max_fpr]
    if under_fpr:
        under_fpr.sort(key=lambda r: (r["overall_correct"], r["bot_correct"]), reverse=True)
        best = under_fpr[0]
        best["mode"] = "best_under_fpr_cap"
        return best

    def score_fn(r):
        return (
            r["overall_correct"]
            + 0.8 * r["bot_correct"]
            - 2.0 * max(0.0, r["fpr"] - max_fpr)
            - 1.5 * max(0.0, min_bot - r["bot_correct"])
            - 1.5 * max(0.0, 0.95 - r["overall_correct"])
        )

    best = max(rows, key=score_fn)
    best["mode"] = "fallback_tradeoff"
    return best


CONFIGS = [
    # Capacity (better recent-day fit when neighbors available)
    dict(max_depth=5, learning_rate=0.05, min_child_weight=2, subsample=0.85,
         colsample_bytree=0.85, reg_lambda=2.0, reg_alpha=0.0, gamma=0.0,
         n_estimators=450, scale_pos_weight=1.2),
    # Balanced
    dict(max_depth=4, learning_rate=0.04, min_child_weight=4, subsample=0.8,
         colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.2, gamma=0.2,
         n_estimators=500, scale_pos_weight=1.0),
    # Conservative (lower FPR bias)
    dict(max_depth=3, learning_rate=0.03, min_child_weight=8, subsample=0.75,
         colsample_bytree=0.7, reg_lambda=10.0, reg_alpha=0.5, gamma=0.5,
         n_estimators=600, scale_pos_weight=0.9),
]


def make_model(seed: int, cfg: dict) -> XGBClassifier:
    cfg = dict(cfg)
    n_estimators = cfg.pop("n_estimators")
    return XGBClassifier(
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
        **cfg,
    )


def lodo_scores(X, y, dates, pool_mask, recent_dates, cfg, seed, half_life) -> np.ndarray:
    scores = np.full(len(y), np.nan)
    for day in recent_dates:
        te = dates == day
        tr = pool_mask & ~te
        w = recency_weights(dates[tr], sorted(set(dates[tr].tolist())), half_life)
        # Up-weight newest train days more for recent focus.
        model = make_model(seed, cfg)
        model.fit(X[tr], y[tr], sample_weight=w, verbose=False)
        scores[te] = model.predict_proba(X[te])[:, 1]
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--recent-val-days", type=int, default=4)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--min-bot-recall", type=float, default=0.95)
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Building features...", flush=True)
    X, y, dates = build_xy(args.examples)
    unique = sorted(set(dates.tolist()))
    holdout_dates = set(unique[-args.holdout_days :])
    recent_dates = unique[-(args.holdout_days + args.recent_val_days) : -args.holdout_days]
    pool_mask = np.array([d not in holdout_dates for d in dates])
    hold_mask = ~pool_mask
    recent_mask = np.array([d in set(recent_dates) for d in dates])

    print(f"recent LODO={recent_dates} holdout={sorted(holdout_dates)}", flush=True)

    best = None
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"Config {i}/{len(CONFIGS)}: {cfg}", flush=True)
        raw = lodo_scores(X, y, dates, pool_mask, recent_dates, cfg, args.seed, args.half_life_days)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw[recent_mask], y[recent_mask])
        cal = iso.predict(raw[recent_mask])
        thr_info = choose_threshold(y[recent_mask], cal, min_bot=args.min_bot_recall, max_fpr=args.max_fpr)
        metrics = eval_at(y[recent_mask], cal, thr_info["threshold"])
        # selection prioritizes goals
        sel = (
            10 * (1 if metrics["overall_correct_percent"] >= 95 else metrics["overall_correct_percent"] / 100)
            + 8 * (1 if metrics["bot_correct_percent"] >= 95 else metrics["bot_correct_percent"] / 100)
            + 6 * (1 if metrics["fpr_percent"] <= args.max_fpr * 100 else max(0, 1 - metrics["fpr_percent"] / 100))
            + 3 * metrics["bot_recall_at_5fpr"]
            + 2 * metrics["ap"]
        )
        cand = {
            "cfg": cfg,
            "raw": raw,
            "iso": iso,
            "thr_info": thr_info,
            "metrics": metrics,
            "sel": sel,
        }
        print(
            f"  -> recent overall={metrics['overall_correct_percent']}% "
            f"bot={metrics['bot_correct_percent']}% fpr={metrics['fpr_percent']}% "
            f"recall@5fpr={metrics['bot_recall_at_5fpr']:.3f} mode={thr_info['mode']}",
            flush=True,
        )
        if best is None or cand["sel"] > best["sel"]:
            best = cand

    assert best is not None
    thr = float(best["thr_info"]["threshold"])
    iso = best["iso"]

    # Final model on full pool
    print("Fitting final model on pool...", flush=True)
    w_pool = recency_weights(dates[pool_mask], sorted(set(dates[pool_mask].tolist())), args.half_life_days)
    final = make_model(args.seed, best["cfg"])
    final.fit(X[pool_mask], y[pool_mask], sample_weight=w_pool, verbose=False)

    def predict_cal(Xq):
        return iso.predict(final.predict_proba(Xq)[:, 1])

    s_all = predict_cal(X)
    s_hold = predict_cal(X[hold_mask])
    s_pool = predict_cal(X[pool_mask])
    oof_cal = np.full(len(y), np.nan)
    oof_cal[recent_mask] = iso.predict(best["raw"][recent_mask])

    per_day = {}
    for day in recent_dates:
        te = dates == day
        per_day[day] = {
            "at_tuned": eval_at(y[te], oof_cal[te], thr),
            "at_0.5": eval_at(y[te], oof_cal[te], 0.5),
        }

    # Secondary operating points for honesty
    thr_fpr5 = choose_threshold(y[recent_mask], oof_cal[recent_mask], min_bot=0.0, max_fpr=0.05)
    thr_bot95 = choose_threshold(y[recent_mask], oof_cal[recent_mask], min_bot=0.95, max_fpr=1.0)

    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-xgb-v2-gen",
        "model_version": "2.2.2-lodo-recent-fpr",
        "framework": "xgboost_only+isotonic",
        "selected_config": best["cfg"],
        "splits": {
            "recent_lodo_dates": list(recent_dates),
            "holdout_dates": sorted(holdout_dates),
            "pool_n": int(pool_mask.sum()),
            "recent_n": int(recent_mask.sum()),
            "holdout_n": int(hold_mask.sum()),
            "labels": {
                "recent": dict(Counter(y[recent_mask].tolist())),
                "holdout": dict(Counter(y[hold_mask].tolist())),
            },
        },
        "threshold": thr,
        "threshold_selection": best["thr_info"],
        "validation_recent_lodo": {
            "at_tuned_threshold": best["metrics"],
            "at_0.5": eval_at(y[recent_mask], oof_cal[recent_mask], 0.5),
            "operating_point_fpr_le_5pct": {
                "threshold_selection": thr_fpr5,
                "metrics": eval_at(y[recent_mask], oof_cal[recent_mask], thr_fpr5["threshold"]),
            },
            "operating_point_bot_ge_95pct": {
                "threshold_selection": thr_bot95,
                "metrics": eval_at(y[recent_mask], oof_cal[recent_mask], thr_bot95["threshold"]),
            },
            "per_day": per_day,
        },
        "holdout_unseen": {
            "at_tuned_threshold": eval_at(y[hold_mask], s_hold, thr),
            "at_0.5": eval_at(y[hold_mask], s_hold, 0.5),
        },
        "all_labeled_final_model": eval_at(y, s_all, thr),
        "pool_final_model": eval_at(y[pool_mask], s_pool, thr),
        "goal_checks": {
            "recent_val_overall_ge_95": best["metrics"]["overall_correct_percent"] >= 95,
            "recent_val_bot_ge_95": best["metrics"]["bot_correct_percent"] >= 95,
            "recent_val_fpr_le_max": best["metrics"]["fpr_percent"] <= args.max_fpr * 100 + 1e-9,
        },
        "note": (
            "Recent validation uses leave-one-day-out (each recent day scored without that day in training). "
            "If joint goals (overall>=95 & bot>=95 & fpr<=5) are impossible under current feature separation, "
            "report includes separate operating points."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final.save_model(args.out_dir / "model.json")
    with (args.out_dir / "model_bundle.pkl").open("wb") as f:
        pickle.dump(
            {
                "model": final,
                "isotonic": iso,
                "threshold": thr,
                "feature_names": FEATURE_NAMES,
                "model_version": report["model_version"],
                "config": best["cfg"],
            },
            f,
        )
    (args.out_dir / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": thr,
                "max_fpr": args.max_fpr,
                "min_bot_recall": args.min_bot_recall,
                "calibration": "isotonic_on_recent_lodo",
                "model_version": report["model_version"],
            },
            indent=2,
        )
        + "\n"
    )
    (args.out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")

    summary = {
        "model_version": report["model_version"],
        "selected_config": best["cfg"],
        "recent_lodo_dates": report["splits"]["recent_lodo_dates"],
        "holdout_dates": report["splits"]["holdout_dates"],
        "threshold": thr,
        "validation_recent_lodo": report["validation_recent_lodo"]["at_tuned_threshold"],
        "operating_point_fpr_le_5pct": report["validation_recent_lodo"]["operating_point_fpr_le_5pct"]["metrics"],
        "operating_point_bot_ge_95pct": report["validation_recent_lodo"]["operating_point_bot_ge_95pct"]["metrics"],
        "holdout_unseen": report["holdout_unseen"],
        "goal_checks": report["goal_checks"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
