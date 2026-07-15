#!/usr/bin/env python3
"""Competitive v3: selective v3 groups + human hard-negatives + LODO + sealed holdout.

Featurizes full selective matrix once, then column-slices for greedy group search.
Hard-neg human chunks only enter training folds, never sealed holdout / LODO eval.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import competitive_schema as schema_comp
from features import selective_schema as schema_sel
from poker44.batch_calibration import apply_batch_safety_topk_v1, apply_threshold_logit_v1
from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_HUMANS = ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
DEFAULT_OUT = ROOT / "models" / "competitive"


def sanitize_chunk(hands: list[dict]) -> list[dict]:
    return [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]


def recency_weights(dates: np.ndarray, unique_sorted: list[str], half_life_days: float) -> np.ndarray:
    idx = {d: i for i, d in enumerate(unique_sorted)}
    max_i = max(len(unique_sorted) - 1, 1)
    ages = np.array([max_i - idx.get(d, 0) for d in dates], dtype=np.float64)
    decay = math.log(2.0) / max(half_life_days, 1e-6)
    w = np.exp(-decay * ages)
    return w / max(w.mean(), 1e-12)


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
        "human_safety": float(detail.get("human_safety_penalty", 0.0)),
    }


def selection_score(m: dict) -> float:
    return (
        0.35 * m["ap"]
        + 0.45 * m["bot_recall_at_5fpr"]
        + 0.15 * m["reward"]
        + 0.05 * (1.0 - m["hard_fpr@0.5"])
    )


def simulate_batch_reward(y, raw_scores, *, batch_size: int = 40, fraction: float = 0.125) -> dict:
    y = np.asarray(y)
    raw_scores = np.asarray(raw_scores, dtype=float)
    final = np.zeros_like(raw_scores)
    for start in range(0, len(y), batch_size):
        sl = slice(start, start + batch_size)
        final[sl] = apply_batch_safety_topk_v1(raw_scores[sl], max_positive_fraction=fraction)
    return eval_suite(y, final, f"topk_frac_{fraction}_bs{batch_size}")


def apply_post(scores, *, calibrator, logit):
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


def tune_cal_and_logit(y, raw, seed: int):
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
    return lr, best_logit, best_m


def make_xgb(seed: int) -> XGBClassifier:
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


def load_labeled(path: Path, feature_names: list[str]):
    Xs, ys, dates = [], [], []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            ex = json.loads(line)
            hands = sanitize_chunk(ex.get("hands") or [])
            feat = schema_sel.extract_chunk_features(hands, feature_names=feature_names)
            Xs.append([float(feat.get(n, 0.0)) for n in feature_names])
            ys.append(int(ex["label"]))
            dates.append(ex["sourceDate"])
            if (i + 1) % 500 == 0:
                print(f"  labeled {i+1}", flush=True)
    return (
        np.asarray(Xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(dates),
    )


def build_human_hardneg_matrix(
    path: Path,
    feature_names: list[str],
    *,
    n_chunks: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading human hands from {path}...", flush=True)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        hands_all = json.load(fh)
    rng = np.random.default_rng(seed)
    clean = []
    for h in hands_all:
        if not isinstance(h, dict):
            continue
        hh = dict(h)
        hh.pop("label", None)
        clean.append(prepare_hand_for_miner(hh))
    if len(clean) < chunk_size:
        raise RuntimeError("Not enough human hands for hard-neg chunks")
    Xs = []
    for i in range(n_chunks):
        idx = rng.choice(len(clean), size=chunk_size, replace=False)
        chunk = [clean[j] for j in idx]
        feat = schema_sel.extract_chunk_features(chunk, feature_names=feature_names)
        Xs.append([float(feat.get(n, 0.0)) for n in feature_names])
        if (i + 1) % 100 == 0:
            print(f"  hardneg chunks {i+1}/{n_chunks}", flush=True)
    return np.asarray(Xs, dtype=np.float64), np.zeros(n_chunks, dtype=np.int64)


def cols_for_groups(full_names: list[str], groups: list[str]) -> list[int]:
    want = set(schema_sel.feature_names_for_groups(groups))
    return [i for i, n in enumerate(full_names) if n in want]


def lodo_oof(
    X,
    y,
    dates,
    pool_mask,
    recent_dates,
    *,
    X_hn: np.ndarray | None,
    y_hn: np.ndarray | None,
    hn_weight: float,
    half_life: float,
    seed: int,
    model_factory=None,
):
    scores = np.full(len(y), np.nan)
    factory = model_factory or make_xgb
    for day in recent_dates:
        te = dates == day
        tr = pool_mask & ~te
        if not np.any(te) or not np.any(tr):
            continue
        Xtr = X[tr]
        ytr = y[tr]
        w = recency_weights(dates[tr], sorted(set(dates[tr].tolist())), half_life)
        if X_hn is not None and len(X_hn):
            Xtr = np.vstack([Xtr, X_hn])
            ytr = np.concatenate([ytr, y_hn])
            w = np.concatenate([w, np.full(len(y_hn), float(hn_weight), dtype=np.float64)])
        model = factory(seed)
        model.fit(Xtr, ytr, sample_weight=w)
        scores[te] = model.predict_proba(X[te])[:, 1]
    recent_set = set(recent_dates)
    mask = (~np.isnan(scores)) & np.array([d in recent_set for d in dates])
    return scores[mask], y[mask]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--humans", type=Path, default=DEFAULT_HUMANS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--recent-val-days", type=int, default=4)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--hardneg-chunks", type=int, default=400)
    ap.add_argument("--hardneg-chunk-size", type=int, default=40)
    ap.add_argument("--hardneg-weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-hardneg", action="store_true")
    args = ap.parse_args()

    all_groups = list(schema_sel.V3_GROUPS.keys())
    full_names = schema_sel.feature_names_for_groups(all_groups)
    print(f"Full selective width={len(full_names)} (comp={len(schema_comp.FEATURE_NAMES)})", flush=True)

    print("Featurizing labeled examples...", flush=True)
    X_full, y, dates = load_labeled(args.examples, full_names)
    unique = sorted(set(dates.tolist()))
    holdout_dates = unique[-args.holdout_days :]
    recent_dates = unique[-(args.holdout_days + args.recent_val_days) : -args.holdout_days]
    hold_set = set(holdout_dates)
    pool_mask = np.array([d not in hold_set for d in dates])
    hold_mask = ~pool_mask
    print(
        f"n={len(y)} holdout={holdout_dates} lodo={recent_dates} "
        f"pool={pool_mask.sum()} hold={hold_mask.sum()}",
        flush=True,
    )

    X_hn_full = y_hn = None
    if not args.skip_hardneg and args.humans.exists():
        X_hn_full, y_hn = build_human_hardneg_matrix(
            args.humans,
            full_names,
            n_chunks=args.hardneg_chunks,
            chunk_size=args.hardneg_chunk_size,
            seed=args.seed,
        )

    def run_groups(groups: list[str], use_hn: bool) -> dict:
        cols = cols_for_groups(full_names, groups)
        names = [full_names[i] for i in cols]
        X = X_full[:, cols]
        X_hn = X_hn_full[:, cols] if (use_hn and X_hn_full is not None) else None
        oof, oy = lodo_oof(
            X,
            y,
            dates,
            pool_mask,
            recent_dates,
            X_hn=X_hn,
            y_hn=y_hn if X_hn is not None else None,
            hn_weight=args.hardneg_weight,
            half_life=args.half_life_days,
            seed=args.seed,
        )
        cal, logit, _ = tune_cal_and_logit(oy, oof, args.seed)
        scored = apply_post(oof, calibrator=cal, logit=logit)
        m = eval_suite(oy, scored, "lodo")
        return {
            "groups": list(groups),
            "feature_names": names,
            "cols": cols,
            "lodo": m,
            "sel": selection_score(m),
            "logit": logit,
            "calibrator": cal,
            "use_hn": use_hn,
        }

    print("=== Greedy feature-group search (with hardneg if available) ===", flush=True)
    use_hn_default = X_hn_full is not None
    best = run_groups([], use_hn_default)
    print(
        f"baseline sel={best['sel']:.4f} reward={best['lodo']['reward']:.4f} "
        f"bot5={best['lodo']['bot_recall_at_5fpr']:.4f} ap={best['lodo']['ap']:.4f}",
        flush=True,
    )
    history = [{"groups": [], "sel": best["sel"], "lodo": best["lodo"], "n_features": len(best["feature_names"])}]
    selected: list[str] = []
    remaining = list(all_groups)
    while remaining:
        cand_best = None
        for g in remaining:
            trial = run_groups(selected + [g], use_hn_default)
            print(
                f"  +{g}: sel={trial['sel']:.4f} reward={trial['lodo']['reward']:.4f} "
                f"bot5={trial['lodo']['bot_recall_at_5fpr']:.4f} ap={trial['lodo']['ap']:.4f}",
                flush=True,
            )
            history.append(
                {
                    "groups": list(trial["groups"]),
                    "sel": trial["sel"],
                    "lodo": trial["lodo"],
                    "n_features": len(trial["feature_names"]),
                }
            )
            if trial["sel"] > best["sel"] + 1e-4 and (cand_best is None or trial["sel"] > cand_best["sel"]):
                cand_best = trial
                cand_best["_added"] = g
        if cand_best is None:
            print("No improving group; stop.", flush=True)
            break
        selected.append(cand_best["_added"])
        remaining.remove(cand_best["_added"])
        best = cand_best
        print(f"KEEP +{cand_best['_added']} -> {selected}", flush=True)

    # Hardneg on/off for chosen groups
    print("=== Hardneg on/off for selected groups ===", flush=True)
    variants = []
    for use_hn in (False, True) if X_hn_full is not None else (False,):
        v = run_groups(selected, use_hn)
        tag = "with_hn" if use_hn else "no_hn"
        print(
            f"  {tag}: sel={v['sel']:.4f} reward={v['lodo']['reward']:.4f} "
            f"bot5={v['lodo']['bot_recall_at_5fpr']:.4f}",
            flush=True,
        )
        variants.append(v)
    variants.sort(key=lambda v: v["sel"], reverse=True)
    chosen = variants[0]
    print(f"Chosen: groups={chosen['groups']} hn={chosen['use_hn']}", flush=True)

    cols = chosen["cols"]
    names = chosen["feature_names"]
    X = X_full[:, cols]
    X_hn = X_hn_full[:, cols] if chosen["use_hn"] and X_hn_full is not None else None

    print("=== Sealed holdout ===", flush=True)
    w_pool = recency_weights(dates[pool_mask], sorted(set(dates[pool_mask].tolist())), args.half_life_days)
    Xtr, ytr, w = X[pool_mask], y[pool_mask], w_pool
    if X_hn is not None:
        Xtr = np.vstack([Xtr, X_hn])
        ytr = np.concatenate([ytr, y_hn])
        w = np.concatenate([w, np.full(len(y_hn), args.hardneg_weight)])
    model_pool = make_xgb(args.seed)
    model_pool.fit(Xtr, ytr, sample_weight=w)
    raw_ho = model_pool.predict_proba(X[hold_mask])[:, 1]
    ho = apply_post(raw_ho, calibrator=chosen["calibrator"], logit=chosen["logit"])
    m_ho = eval_suite(y[hold_mask], ho, "holdout_sealed")
    m_topk = simulate_batch_reward(y[hold_mask], ho)
    print(
        f"SEALED reward={m_ho['reward']:.4f} ap={m_ho['ap']:.4f} "
        f"bot@5fpr={m_ho['bot_recall_at_5fpr']:.4f} fpr={m_ho['hard_fpr@0.5']:.4f}",
        flush=True,
    )

    print("=== Deploy fit on all labeled (+ hardneg) ===", flush=True)
    w_all = recency_weights(dates, unique, args.half_life_days)
    Xd, yd, wd = X, y, w_all
    if X_hn is not None:
        Xd = np.vstack([X, X_hn])
        yd = np.concatenate([y, y_hn])
        wd = np.concatenate([w_all, np.full(len(y_hn), args.hardneg_weight)])
    model = make_xgb(args.seed)
    model.fit(Xd, yd, sample_weight=wd)

    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-competitive-v3-selective",
        "model_version": "6.0.0",
        "selected_groups": selected,
        "n_features": len(names),
        "hardneg": {
            "enabled": bool(chosen["use_hn"]),
            "chunks": args.hardneg_chunks if chosen["use_hn"] else 0,
            "chunk_size": args.hardneg_chunk_size,
            "weight": args.hardneg_weight,
        },
        "score_remap": chosen["logit"],
        "holdout_dates": holdout_dates,
        "recent_lodo_dates": recent_dates,
        "label_counts": {
            "all": dict(Counter(y.tolist())),
            "holdout": dict(Counter(y[hold_mask].tolist())),
        },
        "greedy_history": history,
        "lodo_variants": [
            {
                "groups": v["groups"],
                "use_hn": v["use_hn"],
                "sel": v["sel"],
                "lodo": v["lodo"],
                "logit": v["logit"],
            }
            for v in variants
        ],
        "metrics": {
            "holdout_sealed": m_ho,
            "holdout_sealed_topk": m_topk,
            "lodo_best": chosen["lodo"],
        },
        "selection_policy": (
            "Greedy v3-group add on LODO; hardneg only in train folds; "
            "holdout sealed (incl. newest day); deploy on all labeled"
        ),
        "benchmark_n": int(len(y)),
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
        "score_remap": chosen["logit"] or {},
        "feature_set": "selective",
        "selected_groups": selected,
        "holdout_dates": holdout_dates,
        "trained_at_utc": report["trained_at_utc"],
        "framework": "xgb_selective_v3",
    }

    artifact = {
        "kind": "single",
        "models": [model],
        "weights": [1.0],
        "calibrator": chosen["calibrator"],
        "feature_names": names,
        "feature_set": "selective",
        "metadata": metadata,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "current.joblib"
    joblib.dump(artifact, out_path)
    (args.out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out_dir / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "batch_calibration": "topk_v1",
                "score_remap": chosen["logit"],
                "model_name": report["model_name"],
                "model_version": report["model_version"],
                "selected_groups": selected,
                "model_path": str(out_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"groups": selected, "metrics": report["metrics"], "hardneg": report["hardneg"]}, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
