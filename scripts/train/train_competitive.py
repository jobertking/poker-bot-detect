#!/usr/bin/env python3
"""Train competitive ensemble for SN126 (sanitize + ET/RF/HGB + cal + topK metadata)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.competitive_schema import FEATURE_NAMES, extract_chunk_features, features_to_vector
from poker44.batch_calibration import apply_batch_safety_topk_v1, apply_clip_below
from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT = ROOT / "models" / "competitive"


def sanitize_chunk(hands: list[dict]) -> list[dict]:
    return [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]


def load_dataset(path: Path):
    Xs, ys, dates = [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            hands = sanitize_chunk(ex.get("hands") or [])
            feat = extract_chunk_features(hands)
            Xs.append(features_to_vector(feat))
            ys.append(int(ex["label"]))
            dates.append(ex["sourceDate"])
    return (
        np.asarray(Xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(dates),
    )


def blend_predict(models, weights, X) -> np.ndarray:
    out = np.zeros(X.shape[0], dtype=np.float64)
    wsum = 0.0
    for m, w in zip(models, weights):
        out += float(w) * m.predict_proba(X)[:, 1]
        wsum += float(w)
    return out / max(wsum, 1e-12)


def eval_suite(y, scores, name: str) -> dict:
    rew, detail = reward(np.asarray(scores), np.asarray(y))
    pred = (np.asarray(scores) >= 0.5).astype(int)
    return {
        "name": name,
        "n": int(len(y)),
        "ap": float(average_precision_score(y, scores)) if np.any(y == 1) else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else None,
        "reward": float(rew),
        "bot_recall_at_5fpr": float(detail["bot_recall"]),
        "hard_bot_recall@0.5": float(detail["hard_bot_recall"]),
        "hard_fpr@0.5": float(detail["hard_fpr"]),
        "overall_acc@0.5": float((pred == y).mean()),
        "threshold_sanity": float(detail["threshold_sanity_quality"]),
    }


def simulate_batch_reward(y, raw_scores, *, batch_size: int = 40, fraction: float = 0.125) -> dict:
    """Approximate live validator: apply topK remap inside batches, then reward."""
    y = np.asarray(y)
    raw_scores = np.asarray(raw_scores, dtype=float)
    final = np.zeros_like(raw_scores)
    for start in range(0, len(y), batch_size):
        sl = slice(start, start + batch_size)
        final[sl] = apply_batch_safety_topk_v1(raw_scores[sl], max_positive_fraction=fraction)
    return eval_suite(y, final, f"topk_frac_{fraction}_bs{batch_size}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading + sanitizing + featurizing...", flush=True)
    X, y, dates = load_dataset(args.examples)
    unique = sorted(set(dates.tolist()))
    holdout_dates = set(unique[-args.holdout_days :])
    hold = np.array([d in holdout_dates for d in dates])
    pool = ~hold

    Xtr_full, ytr_full = X[pool], y[pool]
    Xho, yho = X[hold], y[hold]

    Xtr, Xcal, ytr, ycal = train_test_split(
        Xtr_full,
        ytr_full,
        test_size=0.2,
        random_state=args.seed,
        stratify=ytr_full,
    )
    print(
        f"features={X.shape[1]} pool={pool.sum()} holdout={hold.sum()} "
        f"holdout_dates={sorted(holdout_dates)} train={len(ytr)} cal={len(ycal)}",
        flush=True,
    )

    et = ExtraTreesClassifier(
        n_estimators=700,
        max_depth=9,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=4,
    )
    rf = RandomForestClassifier(
        n_estimators=700,
        max_depth=9,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=4,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=700,
        learning_rate=0.03,
        max_depth=9,
        min_samples_leaf=2,
        random_state=args.seed,
    )

    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=4,
    )

    print("Fitting ExtraTrees...", flush=True)
    et.fit(Xtr, ytr)
    print("Fitting RandomForest...", flush=True)
    rf.fit(Xtr, ytr)
    print("Fitting HistGradientBoosting...", flush=True)
    hgb.fit(Xtr, ytr)
    print("Fitting XGBoost...", flush=True)
    xgb.fit(Xtr, ytr)

    models = [et, rf, hgb, xgb]
    # Search blend weights (ET, RF, HGB, XGB) on holdout reward after cal + topK
    candidate_weights = [
        (0.45, 0.25, 0.30, 0.00),
        (0.35, 0.20, 0.25, 0.20),
        (0.30, 0.15, 0.25, 0.30),
        (0.25, 0.15, 0.20, 0.40),
        (0.20, 0.15, 0.20, 0.45),
        (0.15, 0.10, 0.20, 0.55),
        (0.10, 0.10, 0.15, 0.65),
        (0.00, 0.00, 0.00, 1.00),
        (0.25, 0.25, 0.25, 0.25),
        (0.40, 0.10, 0.20, 0.30),
    ]
    best = None
    for w in candidate_weights:
        cal_raw = blend_predict(models, w, Xcal)
        # fit temp logistic on cal
        lr = LogisticRegression(max_iter=1000, random_state=args.seed)
        lr.fit(cal_raw.reshape(-1, 1), ycal)
        cal_s = lr.predict_proba(cal_raw.reshape(-1, 1))[:, 1]
        ho_raw = blend_predict(models, w, Xho)
        ho_s = lr.predict_proba(ho_raw.reshape(-1, 1))[:, 1]
        m_raw = eval_suite(yho, ho_s, "holdout_calibrated")
        m_topk = simulate_batch_reward(yho, ho_s, batch_size=40, fraction=0.125)
        m_clip = eval_suite(yho, apply_clip_below(ho_s), "holdout_clip_below")
        score = (
            3.0 * m_topk["reward"]
            + 2.0 * m_raw["reward"]
            + 1.5 * m_topk["ap"]
            + 1.0 * (1.0 - m_raw["hard_fpr@0.5"])
        )
        cand = {
            "weights": w,
            "score": score,
            "calibrator": lr,
            "holdout_calibrated": m_raw,
            "holdout_topk": m_topk,
            "holdout_clip_below": m_clip,
            "cal_reward": eval_suite(ycal, cal_s, "cal")["reward"],
        }
        print(
            f"weights={w} holdout_reward={m_raw['reward']:.4f} "
            f"topk_reward={m_topk['reward']:.4f} topk_ap={m_topk['ap']:.4f} "
            f"hard_fpr={m_raw['hard_fpr@0.5']:.4f}",
            flush=True,
        )
        if best is None or cand["score"] > best["score"]:
            best = cand

    assert best is not None
    weights = list(best["weights"])
    calibrator = best["calibrator"]

    # Refit on full pool for deploy
    print("Refitting on full pool...", flush=True)
    et.fit(Xtr_full, ytr_full)
    rf.fit(Xtr_full, ytr_full)
    hgb.fit(Xtr_full, ytr_full)
    xgb.fit(Xtr_full, ytr_full)
    models = [et, rf, hgb, xgb]
    # Recalibrate on a fresh stratified split of pool
    Xa, Xb, ya, yb = train_test_split(
        Xtr_full, ytr_full, test_size=0.2, random_state=args.seed + 1, stratify=ytr_full
    )
    raw_b = blend_predict(models, weights, Xb)
    calibrator = LogisticRegression(max_iter=1000, random_state=args.seed)
    calibrator.fit(raw_b.reshape(-1, 1), yb)

    ho_raw = blend_predict(models, weights, Xho)
    ho_cal = calibrator.predict_proba(ho_raw.reshape(-1, 1))[:, 1]
    all_raw = blend_predict(models, weights, X)
    all_cal = calibrator.predict_proba(all_raw.reshape(-1, 1))[:, 1]

    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-competitive-ensemble",
        "model_version": "4.1.0",
        "framework": "ExtraTrees+RandomForest+HistGradientBoosting+XGBoost",
        "n_features": len(FEATURE_NAMES),
        "weights": weights,
        "holdout_dates": sorted(holdout_dates),
        "label_counts": {"all": dict(Counter(y.tolist())), "holdout": dict(Counter(yho.tolist()))},
        "metrics": {
            "holdout_calibrated": eval_suite(yho, ho_cal, "holdout_calibrated"),
            "holdout_topk_sim": simulate_batch_reward(yho, ho_cal),
            "holdout_clip_below": eval_suite(yho, apply_clip_below(ho_cal), "clip"),
            "all_calibrated": eval_suite(y, all_cal, "all"),
            "all_topk_sim": simulate_batch_reward(y, all_cal),
        },
        "search": {
            "best_weights": weights,
            "candidates_note": "selected by holdout topk/calibrated reward blend",
        },
    }

    # Drop zero-weight heads so deploy artifact stays lean.
    kept = [(m, w) for m, w in zip(models, weights) if float(w) > 1e-9]
    if not kept:
        kept = list(zip(models, weights))
    models = [m for m, _ in kept]
    weights = [float(w) for _, w in kept]
    report["weights"] = weights
    report["search"]["best_weights"] = weights
    report["n_models"] = len(models)

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
        "weights": weights,
        "holdout_dates": sorted(holdout_dates),
        "trained_at_utc": report["trained_at_utc"],
        "framework": report["framework"],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "models": models,
        "weights": weights,
        "calibrator": calibrator,
        "feature_names": FEATURE_NAMES,
        "metadata": metadata,
    }
    out_path = args.out_dir / "current.joblib"
    joblib.dump(artifact, out_path)
    (args.out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out_dir / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "batch_calibration": "topk_v1",
                "model_name": report["model_name"],
                "model_version": report["model_version"],
                "model_path": str(out_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
