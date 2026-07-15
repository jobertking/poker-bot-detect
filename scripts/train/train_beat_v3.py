#!/usr/bin/env python3
"""Beat xgb_v3_holdout on sealed 7/13-7/14 via schema blend + hard-day upweight.

Selection uses LODO on days before the sealed block (never peeks at 7/13-14).
Also reports sealed 7/14-15. Deploys winner trained on all labeled data.
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
from xgboost import Booster, DMatrix, XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import beat_v3_schema as schema_beat
from features import chunk_features as schema_v3
from features import competitive_fn_schema as schema_cfn
from poker44.artifact_io import atomic_joblib_dump, atomic_write_text, prune_archive, recipe_fingerprint
from poker44.validator.payload_view import prepare_hand_for_miner
from scripts.train.train_competitive_v3 import (
    apply_post,
    eval_suite,
    make_xgb,
    recency_weights,
    selection_score,
    simulate_batch_reward,
    tune_cal_and_logit,
)

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT = ROOT / "models" / "competitive"
XGB_V3_MODEL = ROOT / "models" / "xgb_v3_holdout" / "model.json"
XGB_V3_NPZ = ROOT / "models" / "xgb_v3_holdout" / "features_used.npz"


def sanitize(hands):
    return [prepare_hand_for_miner(h) for h in (hands or []) if isinstance(h, dict)]


def load_matrix(path: Path, schema_mod):
    names = list(schema_mod.FEATURE_NAMES)
    Xs, ys, dates = [], [], []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            ex = json.loads(line)
            feat = schema_mod.extract_chunk_features(sanitize(ex.get("hands") or []))
            Xs.append(schema_mod.features_to_vector(feat))
            ys.append(int(ex["label"]))
            dates.append(ex["sourceDate"])
            if (i + 1) % 500 == 0:
                print(f"  {schema_mod.__name__} {i+1}", flush=True)
    return (
        np.asarray(Xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(dates),
        names,
    )


def hard_day_weights(
    dates: np.ndarray,
    unique_sorted: list[str],
    half_life: float,
    hard_days: set[str],
    hard_mult: float,
) -> np.ndarray:
    w = recency_weights(dates, unique_sorted, half_life)
    if hard_days and hard_mult != 1.0:
        boost = np.array([hard_mult if d in hard_days else 1.0 for d in dates], dtype=np.float64)
        w = w * boost
        w = w / max(w.mean(), 1e-12)
    return w


def make_xgb_cfg(seed: int, cfg: str) -> XGBClassifier:
    if cfg == "v3ish":
        # Closer to earlier generalize/capacity settings that ranked well.
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
    return make_xgb(seed)


def lodo_oof_weighted(
    X,
    y,
    dates,
    pool_mask,
    recent_dates,
    *,
    half_life: float,
    hard_days: set[str],
    hard_mult: float,
    seed: int,
    cfg: str,
):
    scores = np.full(len(y), np.nan)
    for day in recent_dates:
        te = dates == day
        tr = pool_mask & ~te
        if not np.any(te) or not np.any(tr):
            continue
        w = hard_day_weights(
            dates[tr],
            sorted(set(dates[tr].tolist())),
            half_life,
            hard_days,
            hard_mult,
        )
        model = make_xgb_cfg(seed, cfg)
        model.fit(X[tr], y[tr], sample_weight=w)
        scores[te] = model.predict_proba(X[te])[:, 1]
    recent_set = set(recent_dates)
    mask = (~np.isnan(scores)) & np.array([d in recent_set for d in dates])
    return scores[mask], y[mask]


def eval_xgb_v3_baseline() -> dict:
    npz = np.load(XGB_V3_NPZ, allow_pickle=True)
    X = npz["X"]
    y = npz["y"].astype(int)
    dates = npz["sourceDates"]
    bst = Booster()
    bst.load_model(str(XGB_V3_MODEL))
    scores = bst.predict(DMatrix(X.astype(np.float32)))
    hold = np.array([d in ("2026-07-13", "2026-07-14") for d in dates])
    return eval_suite(y[hold], scores[hold], "xgb_v3_native_7/13-14")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading feature matrices...", flush=True)
    mats = {}
    for key, mod in [
        ("cfn", schema_cfn),
        ("beat", schema_beat),
        ("v3", schema_v3),
    ]:
        print(f"  schema={key}", flush=True)
        X, y, dates, names = load_matrix(args.examples, mod)
        mats[key] = (X, y, dates, names)

    y = mats["cfn"][1]
    dates = mats["cfn"][2]
    assert np.array_equal(y, mats["beat"][1]) and np.array_equal(dates, mats["beat"][2])

    unique = sorted(set(dates.tolist()))
    # Primary sealed = xgb_v3 holdout days
    sealed_a = ["2026-07-13", "2026-07-14"]
    sealed_b = ["2026-07-14", "2026-07-15"]
    # LODO selection days: 4 days before 7/13
    idx_a = unique.index("2026-07-13")
    recent = unique[max(0, idx_a - 4) : idx_a]
    hold_a = set(sealed_a)
    pool_a = np.array([d not in hold_a for d in dates])
    print(f"LODO select days={recent} sealed_primary={sealed_a}", flush=True)

    print("xgb_v3 baseline...", flush=True)
    base_v3 = eval_xgb_v3_baseline()
    print(
        f"  xgb_v3: reward={base_v3['reward']:.4f} ap={base_v3['ap']:.4f} "
        f"bot@5={base_v3['bot_recall_at_5fpr']:.4f}",
        flush=True,
    )

    # Hard days from earlier FN analysis within selection window vicinity
    hard_candidates = [
        set(),
        {"2026-07-12"},
        {"2026-07-11", "2026-07-12"},
    ]
    hard_mults = [1.0, 2.5]

    candidates = []
    cfgs = [
        ("cfn", "reg", mats["cfn"]),
        ("cfn", "v3ish", mats["cfn"]),
        ("beat", "reg", mats["beat"]),
        ("beat", "v3ish", mats["beat"]),
        ("v3", "reg", mats["v3"]),
        ("v3", "v3ish", mats["v3"]),
    ]

    for schema_key, cfg, (X, _, _, names) in cfgs:
        for hard_days in hard_candidates:
            for hm in hard_mults:
                if not hard_days and hm != 1.0:
                    continue
                tag = f"{schema_key}|{cfg}|hard={sorted(hard_days) or '-'}|x{hm}"
                print(f"\n=== LODO {tag} ===", flush=True)
                oof, oy = lodo_oof_weighted(
                    X,
                    y,
                    dates,
                    pool_a,
                    recent,
                    half_life=args.half_life_days,
                    hard_days=hard_days,
                    hard_mult=hm,
                    seed=args.seed,
                    cfg=cfg,
                )
                cal, logit, _ = tune_cal_and_logit(oy, oof, args.seed)
                scored = apply_post(oof, calibrator=cal, logit=logit)
                m = eval_suite(oy, scored, "lodo")
                sel = selection_score(m)
                print(
                    f"  LODO reward={m['reward']:.4f} bot@5={m['bot_recall_at_5fpr']:.4f} "
                    f"ap={m['ap']:.4f} sel={sel:.4f}",
                    flush=True,
                )
                candidates.append(
                    {
                        "tag": tag,
                        "schema_key": schema_key,
                        "cfg": cfg,
                        "hard_days": sorted(hard_days),
                        "hard_mult": hm,
                        "names": names,
                        "X": X,
                        "cal": cal,
                        "logit": logit,
                        "lodo": m,
                        "sel": sel,
                    }
                )

    # Also try score-blend of best cfn and best v3 after LODO (on OOF)
    # Rebuild OOF full-index for top schemas quickly under best hard settings so far
    candidates.sort(key=lambda c: c["sel"], reverse=True)
    print("\nTop5 LODO:", flush=True)
    for c in candidates[:5]:
        print(
            f"  {c['tag']}: sel={c['sel']:.4f} reward={c['lodo']['reward']:.4f} "
            f"bot@5={c['lodo']['bot_recall_at_5fpr']:.4f}",
            flush=True,
        )

    def sealed_for(cand, sealed_days: list[str]) -> dict:
        hold_set = set(sealed_days)
        pool = np.array([d not in hold_set for d in dates])
        hold = ~pool
        X = cand["X"]
        w = hard_day_weights(
            dates[pool],
            sorted(set(dates[pool].tolist())),
            args.half_life_days,
            set(cand["hard_days"]),
            cand["hard_mult"],
        )
        model = make_xgb_cfg(args.seed, cand["cfg"])
        model.fit(X[pool], y[pool], sample_weight=w)
        raw = model.predict_proba(X[hold])[:, 1]
        post = apply_post(raw, calibrator=cand["cal"], logit=cand["logit"])
        return eval_suite(y[hold], post, f"sealed_{sealed_days[0]}_{sealed_days[-1]}")

    # Evaluate top LODO candidates on sealed days for reporting only (never for selection).
    print("\n=== Sealed primary 7/13-14 (report only; selection uses LODO) ===", flush=True)
    for cand in candidates[:8]:
        m_seal = sealed_for(cand, sealed_a)
        cand["sealed_713_714"] = m_seal
        beat = (
            m_seal["reward"] > base_v3["reward"]
            and m_seal["bot_recall_at_5fpr"] >= base_v3["bot_recall_at_5fpr"] - 0.005
        )
        print(
            f"  {cand['tag']}: reward={m_seal['reward']:.4f} bot@5={m_seal['bot_recall_at_5fpr']:.4f} "
            f"ap={m_seal['ap']:.4f} fpr={m_seal['hard_fpr@0.5']:.4f} beat_v3={beat}",
            flush=True,
        )

    # Deploy pick: best LODO selection_score among candidates (no sealed peek).
    candidates.sort(key=lambda c: (c["sel"], c["lodo"]["reward"], c["lodo"]["bot_recall_at_5fpr"]), reverse=True)
    best = candidates[0]
    pick_reason = "best_lodo_selection_score"
    best["sealed_714_715"] = sealed_for(best, sealed_b)
    print(f"\n>>> SELECTED {best['tag']} ({pick_reason})", flush=True)
    print(
        f"  sealed 7/13-14: {best['sealed_713_714']['reward']:.4f} / bot@5={best['sealed_713_714']['bot_recall_at_5fpr']:.4f}",
        flush=True,
    )
    print(
        f"  sealed 7/14-15: {best['sealed_714_715']['reward']:.4f} / bot@5={best['sealed_714_715']['bot_recall_at_5fpr']:.4f}",
        flush=True,
    )
    print(
        f"  xgb_v3 baseline: {base_v3['reward']:.4f} / bot@5={base_v3['bot_recall_at_5fpr']:.4f}",
        flush=True,
    )

    # Deploy on all dates
    print("Deploy fit on all labeled...", flush=True)
    X = best["X"]
    w_all = hard_day_weights(
        dates,
        unique,
        args.half_life_days,
        set(best["hard_days"]),
        best["hard_mult"],
    )
    model = make_xgb_cfg(args.seed, best["cfg"])
    model.fit(X, y, sample_weight=w_all)

    feature_set = {
        "cfn": "competitive_fn",
        "beat": "beat_v3",
        "v3": "v3",
    }[best["schema_key"]]

    # register beat_v3 schema in inference if needed
    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-beat-v3",
        "model_version": "7.0.0",
        "selected": best["tag"],
        "pick_reason": pick_reason,
        "feature_set": feature_set,
        "n_features": len(best["names"]),
        "cfg": best["cfg"],
        "hard_days": best["hard_days"],
        "hard_mult": best["hard_mult"],
        "score_remap": best["logit"],
        "xgb_v3_baseline_713_714": base_v3,
        "lodo_dates": recent,
        "metrics": {
            "lodo": best["lodo"],
            "sealed_713_714": best["sealed_713_714"],
            "sealed_714_715": best["sealed_714_715"],
            "sealed_713_714_topk": simulate_batch_reward(
                y[np.array([d in hold_a for d in dates])],
                # reuse sealed scores approximately via re-fit already done; skip exact topk recompute blob
                apply_post(
                    make_xgb_cfg(args.seed, best["cfg"])
                    .fit(
                        X[pool_a],
                        y[pool_a],
                        sample_weight=hard_day_weights(
                            dates[pool_a],
                            sorted(set(dates[pool_a].tolist())),
                            args.half_life_days,
                            set(best["hard_days"]),
                            best["hard_mult"],
                        ),
                    )
                    .predict_proba(X[np.array([d in hold_a for d in dates])])[:, 1],
                    calibrator=best["cal"],
                    logit=best["logit"],
                ),
            ),
        },
        "lodo_top5": [
            {
                "tag": c["tag"],
                "sel": c["sel"],
                "lodo": c["lodo"],
                "sealed_713_714": c.get("sealed_713_714"),
            }
            for c in candidates[:5]
        ],
        "benchmark_n": int(len(y)),
        "label_counts": {"all": dict(Counter(y.tolist()))},
        "beats_xgb_v3_on_713_714": bool(
            best["sealed_713_714"]["reward"] > base_v3["reward"]
            and best["sealed_713_714"]["bot_recall_at_5fpr"]
            >= base_v3["bot_recall_at_5fpr"] - 0.005
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
        "feature_set": feature_set,
        "trained_at_utc": report["trained_at_utc"],
        "framework": f"xgb_{feature_set}_{best['cfg']}",
    }
    artifact = {
        "kind": "single",
        "models": [model],
        "weights": [1.0],
        "calibrator": best["cal"],
        "feature_names": best["names"],
        "feature_set": feature_set,
        "metadata": metadata,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "current.joblib"
    # archive previous
    if out_path.exists():
        arch = args.out_dir / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(out_path, arch / f"current_{stamp}.joblib")
        prune_archive(arch, keep=int(os.getenv("POKER44_ARCHIVE_KEEP", "30")))

    report["recipe_fingerprint"] = recipe_fingerprint(ROOT)
    atomic_joblib_dump(artifact, out_path)
    # strip huge blobs from report candidates
    atomic_write_text(args.out_dir / "train_report.json", json.dumps(report, indent=2) + "\n")
    atomic_write_text(
        args.out_dir / "threshold.json",
        json.dumps(
            {
                "threshold": 0.5,
                "batch_calibration": "topk_v1",
                "score_remap": best["logit"],
                "model_name": report["model_name"],
                "model_version": report["model_version"],
                "feature_set": feature_set,
                "beats_xgb_v3_on_713_714": report["beats_xgb_v3_on_713_714"],
                "recipe_fingerprint": report["recipe_fingerprint"],
                "model_path": str(out_path),
            },
            indent=2,
        )
        + "\n",
    )
    print(json.dumps({k: report[k] for k in ("selected", "pick_reason", "beats_xgb_v3_on_713_714", "metrics")}, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
