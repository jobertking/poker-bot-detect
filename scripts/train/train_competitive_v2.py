#!/usr/bin/env python3
"""Competitive v2: LODO selection, sealed holdout, merged features, logit remap.

Selection never peeks at the final holdout days. Candidates compared on
recent-date LODO reward (AP + recall@5fpr focused). Final report is one-shot
holdout eval.
"""

from __future__ import annotations

import argparse
import json
import math
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
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import competitive_schema as schema_comp
from features import merged_schema as schema_merged
from poker44.batch_calibration import (
    apply_batch_safety_topk_v1,
    apply_threshold_logit_v1,
)
from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT = ROOT / "models" / "competitive"


def sanitize_chunk(hands: list[dict]) -> list[dict]:
    return [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]


def load_dataset(path: Path, schema_mod):
    Xs, ys, dates = [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            hands = sanitize_chunk(ex.get("hands") or [])
            feat = schema_mod.extract_chunk_features(hands)
            Xs.append(schema_mod.features_to_vector(feat))
            ys.append(int(ex["label"]))
            dates.append(ex["sourceDate"])
    return (
        np.asarray(Xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(dates),
        list(schema_mod.FEATURE_NAMES),
    )


def recency_weights(dates: np.ndarray, unique_sorted: list[str], half_life_days: float) -> np.ndarray:
    idx = {d: i for i, d in enumerate(unique_sorted)}
    max_i = max(len(unique_sorted) - 1, 1)
    ages = np.array([max_i - idx[d] for d in dates], dtype=np.float64)
    decay = math.log(2.0) / max(half_life_days, 1e-6)
    w = np.exp(-decay * ages)
    return w / max(w.mean(), 1e-12)


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
        "human_safety": float(detail.get("human_safety_penalty", detail.get("threshold_sanity_quality", 0.0))),
    }


def simulate_batch_reward(y, raw_scores, *, batch_size: int = 40, fraction: float = 0.125) -> dict:
    y = np.asarray(y)
    raw_scores = np.asarray(raw_scores, dtype=float)
    final = np.zeros_like(raw_scores)
    for start in range(0, len(y), batch_size):
        sl = slice(start, start + batch_size)
        final[sl] = apply_batch_safety_topk_v1(raw_scores[sl], max_positive_fraction=fraction)
    return eval_suite(y, final, f"topk_frac_{fraction}_bs{batch_size}")


def selection_score(m: dict) -> float:
    """Focus on ranking terms that still have headroom."""
    return (
        0.35 * m["ap"]
        + 0.45 * m["bot_recall_at_5fpr"]
        + 0.15 * m["reward"]
        + 0.05 * (1.0 - m["hard_fpr@0.5"])
    )


def make_xgb(seed: int, *, regularized: bool = True) -> XGBClassifier:
    if regularized:
        return XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=4,
            reg_lambda=6.0,
            reg_alpha=0.3,
            gamma=0.2,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
        )
    return XGBClassifier(
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
        random_state=seed,
        n_jobs=4,
    )


def make_tree_trio(seed: int):
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=9,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=4,
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=9,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=4,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.03,
        max_depth=9,
        min_samples_leaf=2,
        random_state=seed,
    )
    return et, rf, hgb


def fit_candidate(kind: str, X, y, w, seed: int):
    if kind == "xgb":
        model = make_xgb(seed, regularized=True)
        model.fit(X, y, sample_weight=w)
        return [model], [1.0]
    if kind == "xgb_cap":
        model = make_xgb(seed, regularized=False)
        model.fit(X, y, sample_weight=w)
        return [model], [1.0]
    et, rf, hgb = make_tree_trio(seed)
    et.fit(X, y, sample_weight=w)
    rf.fit(X, y, sample_weight=w)
    hgb.fit(X, y, sample_weight=w)
    return [et, rf, hgb], [0.45, 0.25, 0.30]


def apply_post(
    scores: np.ndarray,
    *,
    calibrator,
    logit: dict | None,
) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if calibrator is not None and s.size:
        s = calibrator.predict_proba(s.reshape(-1, 1))[:, 1]
    if logit:
        s = np.asarray(
            apply_threshold_logit_v1(
                s,
                threshold=float(logit["threshold"]),
                temperature=float(logit["temperature"]),
            ),
            dtype=float,
        )
    return s


def tune_cal_and_logit(y, raw, seed: int) -> tuple[object, dict | None, dict]:
    """Fit LR calibrator; optionally tune threshold_logit on same OOF raw scores."""
    y = np.asarray(y)
    raw = np.asarray(raw, dtype=float)
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(raw.reshape(-1, 1), y)
    cal = lr.predict_proba(raw.reshape(-1, 1))[:, 1]
    base = eval_suite(y, cal, "cal_only")

    best_logit = None
    best_m = base
    best_sel = selection_score(base)
    for thr in (0.40, 0.44, 0.46, 0.48, 0.50, 0.52):
        for temp in (0.05, 0.08, 0.12, 0.18, 0.25):
            rem = np.asarray(apply_threshold_logit_v1(cal, threshold=thr, temperature=temp))
            m = eval_suite(y, rem, f"logit_{thr}_{temp}")
            sel = selection_score(m)
            if sel > best_sel:
                best_sel = sel
                best_m = m
                best_logit = {"kind": "threshold_logit_v1", "threshold": thr, "temperature": temp}
    return lr, best_logit, {"cal_only": base, "best": best_m, "logit": best_logit}


def lodo_oof(
    X,
    y,
    dates,
    pool_mask,
    recent_dates,
    *,
    kind: str,
    half_life: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (oof_raw_scores on recent days, y_recent aligned)."""
    scores = np.full(len(y), np.nan)
    for day in recent_dates:
        te = dates == day
        tr = pool_mask & ~te
        if not np.any(te) or not np.any(tr):
            continue
        w = recency_weights(dates[tr], sorted(set(dates[tr].tolist())), half_life)
        models, weights = fit_candidate(kind, X[tr], y[tr], w, seed)
        scores[te] = blend_predict(models, weights, X[te])
    mask = ~np.isnan(scores) & pool_mask
    # Restrict to recent days only for selection metrics
    recent_set = set(recent_dates)
    mask = mask & np.array([d in recent_set for d in dates])
    return scores[mask], y[mask]


CANDIDATES = [
    ("comp_xgb", "competitive", "xgb"),
    ("comp_xgb_cap", "competitive", "xgb_cap"),
    ("comp_trio", "competitive", "trio"),
    ("merged_xgb", "merged", "xgb"),
    ("merged_xgb_cap", "merged", "xgb_cap"),
    ("merged_trio", "merged", "trio"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--recent-val-days", type=int, default=4)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading competitive features...", flush=True)
    Xc, y, dates, names_comp = load_dataset(args.examples, schema_comp)
    print("Loading merged features...", flush=True)
    Xm, y2, dates2, names_merged = load_dataset(args.examples, schema_merged)
    assert np.array_equal(y, y2) and np.array_equal(dates, dates2)

    unique = sorted(set(dates.tolist()))
    holdout_dates = unique[-args.holdout_days :]
    recent_dates = unique[-(args.holdout_days + args.recent_val_days) : -args.holdout_days]
    holdout_set = set(holdout_dates)
    pool_mask = np.array([d not in holdout_set for d in dates])
    hold_mask = ~pool_mask
    print(
        f"holdout={holdout_dates} (SEALED) recent_lodo={recent_dates} "
        f"pool={pool_mask.sum()} hold={hold_mask.sum()} "
        f"n_comp={Xc.shape[1]} n_merged={Xm.shape[1]}",
        flush=True,
    )

    feature_sets = {
        "competitive": (Xc, names_comp),
        "merged": (Xm, names_merged),
    }

    results = []
    oof_cache = {}
    for name, feat_key, kind in CANDIDATES:
        X, feat_names = feature_sets[feat_key]
        print(f"\n=== LODO select: {name} ({kind} on {feat_key}) ===", flush=True)
        oof_raw, oof_y = lodo_oof(
            X,
            y,
            dates,
            pool_mask,
            recent_dates,
            kind=kind,
            half_life=args.half_life_days,
            seed=args.seed,
        )
        cal, logit, tune_info = tune_cal_and_logit(oof_y, oof_raw, args.seed)
        scored = apply_post(oof_raw, calibrator=cal, logit=logit)
        m = eval_suite(oof_y, scored, "lodo")
        m_topk = simulate_batch_reward(oof_y, scored)
        sel = selection_score(m)
        print(
            f"  LODO reward={m['reward']:.4f} ap={m['ap']:.4f} "
            f"bot@5fpr={m['bot_recall_at_5fpr']:.4f} hard_fpr={m['hard_fpr@0.5']:.4f} "
            f"topk_reward={m_topk['reward']:.4f} sel={sel:.4f} logit={logit}",
            flush=True,
        )
        oof_cache[name] = (oof_raw, oof_y, cal, logit)
        results.append(
            {
                "name": name,
                "feat_key": feat_key,
                "kind": kind,
                "feature_names": feat_names,
                "lodo": m,
                "lodo_topk": m_topk,
                "selection_score": sel,
                "logit": logit,
                "tune": {"cal_only": tune_info["cal_only"], "best": tune_info["best"]},
            }
        )

    results.sort(key=lambda r: r["selection_score"], reverse=True)
    best = results[0]
    print(
        f"\n>>> Selected {best['name']} sel={best['selection_score']:.4f} "
        f"(holdout still sealed)",
        flush=True,
    )

    X, feat_names = feature_sets[best["feat_key"]]
    Xtr_full, ytr_full = X[pool_mask], y[pool_mask]
    Xho, yho = X[hold_mask], y[hold_mask]
    oof_raw, oof_y, calibrator, logit = oof_cache[best["name"]]

    # Honest sealed eval: pool-trained models + LODO-fitted calibrator/logit
    print("Fitting pool models for sealed holdout eval...", flush=True)
    w_pool = recency_weights(
        dates[pool_mask], sorted(set(dates[pool_mask].tolist())), args.half_life_days
    )
    models_pool, weights = fit_candidate(best["kind"], Xtr_full, ytr_full, w_pool, args.seed)
    raw_ho = blend_predict(models_pool, weights, Xho)
    ho_post = apply_post(raw_ho, calibrator=calibrator, logit=logit)
    m_ho = eval_suite(yho, ho_post, "holdout_sealed")
    m_ho_topk = simulate_batch_reward(yho, ho_post)
    m_ho_nologit = eval_suite(
        yho, apply_post(raw_ho, calibrator=calibrator, logit=None), "holdout_cal"
    )
    print(
        f"SEALED holdout: reward={m_ho['reward']:.4f} ap={m_ho['ap']:.4f} "
        f"bot@5fpr={m_ho['bot_recall_at_5fpr']:.4f} fpr={m_ho['hard_fpr@0.5']:.4f} "
        f"topk={m_ho_topk['reward']:.4f} (cal-only reward={m_ho_nologit['reward']:.4f})",
        flush=True,
    )

    # Production fit: include ALL labeled days (holdout too) for live strength.
    # Sealed metrics above remain the honest generalization report.
    print("Refitting on ALL labeled data for deploy artifact...", flush=True)
    w_all = recency_weights(dates, unique, args.half_life_days)
    models, weights = fit_candidate(best["kind"], X, y, w_all, args.seed)

    # Keep LODO calibrator + logit (not fitted on holdout labels via scores).
    all_post = apply_post(blend_predict(models, weights, X), calibrator=calibrator, logit=logit)
    m_all = eval_suite(y, all_post, "all_incl_holdout_report_only")

    kept = [(m, w) for m, w in zip(models, weights) if float(w) > 1e-9]
    models = [m for m, _ in kept]
    weights = [float(w) for _, w in kept]

    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-competitive-v2",
        "model_version": "5.0.0",
        "selected_candidate": best["name"],
        "framework": best["kind"],
        "feature_set": best["feat_key"],
        "n_features": len(feat_names),
        "weights": weights,
        "half_life_days": args.half_life_days,
        "score_remap": logit,
        "holdout_dates": holdout_dates,
        "recent_lodo_dates": recent_dates,
        "label_counts": {
            "all": dict(Counter(y.tolist())),
            "holdout": dict(Counter(yho.tolist())),
        },
        "lodo_ranking": [
            {
                "name": r["name"],
                "selection_score": r["selection_score"],
                "lodo": r["lodo"],
                "lodo_topk": r["lodo_topk"],
                "logit": r["logit"],
            }
            for r in results
        ],
        "metrics": {
            "holdout_sealed": m_ho,
            "holdout_sealed_topk": m_ho_topk,
            "holdout_sealed_cal_only": m_ho_nologit,
            "all_incl_holdout_report_only": m_all,
        },
        "selection_policy": (
            "LODO on recent_val_days; holdout sealed for report; "
            "deploy trains on all dates; calibrator from LODO OOF"
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
        "weights": weights,
        "feature_set": best["feat_key"],
        "holdout_dates": holdout_dates,
        "trained_at_utc": report["trained_at_utc"],
        "framework": best["kind"],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "models": models,
        "weights": weights,
        "calibrator": calibrator,
        "feature_names": feat_names,
        "feature_set": best["feat_key"],
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
                "score_remap": logit,
                "model_name": report["model_name"],
                "model_version": report["model_version"],
                "selected_candidate": best["name"],
                "model_path": str(out_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("selected_candidate", "metrics", "score_remap", "lodo_ranking")
            },
            indent=2,
        )
    )
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
