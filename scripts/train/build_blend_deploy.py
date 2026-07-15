#!/usr/bin/env python3
"""Build deploy blend: 0.2*comp_xgb + 0.8*merged_xgb_cap (LODO-chosen)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import competitive_schema as schema_comp
from features import merged_schema as schema_merged
from poker44.artifact_io import atomic_joblib_dump
from scripts.train.train_competitive_v2 import (
    apply_post,
    eval_suite,
    fit_candidate,
    blend_predict,
    load_dataset,
    lodo_oof,
    recency_weights,
    selection_score,
    simulate_batch_reward,
    tune_cal_and_logit,
)

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
OUT = ROOT / "models" / "competitive"


def main() -> int:
    print("Loading features...", flush=True)
    Xc, y, dates, names_c = load_dataset(DEFAULT_EXAMPLES, schema_comp)
    Xm, _, _, names_m = load_dataset(DEFAULT_EXAMPLES, schema_merged)
    unique = sorted(set(dates.tolist()))
    holdout_dates = unique[-2:]
    recent = unique[-6:-2]
    hold_set = set(holdout_dates)
    pool = np.array([d not in hold_set for d in dates])
    hold = ~pool

    print("LODO parents...", flush=True)
    raw_c, y_c = lodo_oof(Xc, y, dates, pool, recent, kind="xgb", half_life=6.0, seed=42)
    raw_m, _ = lodo_oof(Xm, y, dates, pool, recent, kind="xgb_cap", half_life=6.0, seed=42)

    best = None
    for w in np.linspace(0, 1, 11):
        raw = w * raw_c + (1.0 - w) * raw_m
        cal, logit, _ = tune_cal_and_logit(y_c, raw, 42)
        scored = apply_post(raw, calibrator=cal, logit=logit)
        m = eval_suite(y_c, scored, "lodo")
        sel = selection_score(m)
        print(
            f"w_comp={w:.1f} reward={m['reward']:.4f} ap={m['ap']:.4f} "
            f"bot5={m['bot_recall_at_5fpr']:.4f} sel={sel:.4f}",
            flush=True,
        )
        if best is None or sel > best["sel"]:
            best = {
                "w": float(w),
                "cal": cal,
                "logit": logit,
                "m": m,
                "sel": sel,
            }

    assert best is not None
    w = best["w"]
    print(f"Selected w_comp={w} sel={best['sel']:.4f}", flush=True)

    # Sealed eval
    w_pool = recency_weights(dates[pool], sorted(set(dates[pool].tolist())), 6.0)
    mc, wc = fit_candidate("xgb", Xc[pool], y[pool], w_pool, 42)
    mm, wm = fit_candidate("xgb_cap", Xm[pool], y[pool], w_pool, 42)
    raw_ho = w * blend_predict(mc, wc, Xc[hold]) + (1.0 - w) * blend_predict(mm, wm, Xm[hold])
    ho = apply_post(raw_ho, calibrator=best["cal"], logit=best["logit"])
    m_ho = eval_suite(y[hold], ho, "holdout_sealed")
    m_topk = simulate_batch_reward(y[hold], ho)
    print(
        f"SEALED reward={m_ho['reward']:.4f} ap={m_ho['ap']:.4f} "
        f"bot@5fpr={m_ho['bot_recall_at_5fpr']:.4f}",
        flush=True,
    )

    # Deploy on all data
    print("Fitting deploy heads on all dates...", flush=True)
    w_all = recency_weights(dates, unique, 6.0)
    mc_all, wc_all = fit_candidate("xgb", Xc, y, w_all, 42)
    mm_all, wm_all = fit_candidate("xgb_cap", Xm, y, w_all, 42)

    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-competitive-v2-blend",
        "model_version": "5.1.0",
        "selected_candidate": f"blend_comp{w:.1f}_merged{1-w:.1f}",
        "blend_weight_competitive": w,
        "blend_weight_merged": 1.0 - w,
        "score_remap": best["logit"],
        "holdout_dates": holdout_dates,
        "recent_lodo_dates": recent,
        "label_counts": {
            "all": dict(Counter(y.tolist())),
            "holdout": dict(Counter(y[hold].tolist())),
        },
        "lodo_blend": best["m"],
        "metrics": {
            "holdout_sealed": m_ho,
            "holdout_sealed_topk": m_topk,
        },
        "selection_policy": (
            "Blend weight + cal/logit from LODO (7/9-7/12); "
            "holdout sealed; deploy fits on all dates"
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
        "score_remap": best["logit"] or {},
        "artifact_kind": "blend_v1",
        "holdout_dates": holdout_dates,
        "trained_at_utc": report["trained_at_utc"],
    }

    artifact = {
        "kind": "blend_v1",
        "calibrator": best["cal"],
        "metadata": metadata,
        "heads": [
            {
                "name": "competitive_xgb",
                "feature_set": "competitive",
                "feature_names": names_c,
                "models": mc_all,
                "weights": wc_all,
                "blend_weight": w,
            },
            {
                "name": "merged_xgb_cap",
                "feature_set": "merged",
                "feature_names": names_m,
                "models": mm_all,
                "weights": wm_all,
                "blend_weight": 1.0 - w,
            },
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "current.joblib"
    atomic_joblib_dump(artifact, out_path)
    (OUT / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "batch_calibration": "topk_v1",
                "score_remap": best["logit"],
                "model_name": report["model_name"],
                "model_version": report["model_version"],
                "selected_candidate": report["selected_candidate"],
                "model_path": str(out_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
