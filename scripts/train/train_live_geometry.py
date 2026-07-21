#!/usr/bin/env python3
"""Live-geometry trainer: fair gate + live-cal + traffic-matched aug.

Pipeline:
  1. Audit request logs → drop hand-count / passivity OOD features.
  2. Augment pool to match live hand-count mix (~80% 80–100 hands).
  3. Fit model on pool ONLY (no holdout dates in tree weights).
  4. Calibrate on a live-shaped holdout slice (not LODO).
  5. Gate on a disjoint live-shaped slice (topk100 + topk120).
  6. Deploy the SAME pool model that was gated (no all-data refit).

Default --out-dir is models/staging_live_geometry (live miner untouched until deploy).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import beat_v3_coherent_live_schema as schema
from features import beat_v3_coherent_schema as full_schema
from features import live_robust
from poker44.artifact_io import (
    atomic_joblib_dump,
    atomic_write_text,
    prune_archive,
    recipe_fingerprint,
    read_json,
)
from poker44.large_chunk_augment import (
    LargeAugmentationConfig,
    build_live_distribution_holdout,
    build_training_views,
)
from poker44.miner_inference import CompetitiveMinerModel
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
DEFAULT_OUT = ROOT / "models" / "staging_live_geometry"
COMPETITIVE_DIR = ROOT / "models" / "competitive"
REQUEST_LOG_DIR = ROOT / "logs" / "requests"
MODEL_VERSION = "8.3.0-live-fair"


def make_xgb(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.7,
        min_child_weight=2,
        reg_lambda=2.5,
        reg_alpha=0.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )


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


def resolve_feature_names(args, log_dir: Path) -> tuple[list[str], dict]:
    base_names = list(full_schema.FEATURE_NAMES)
    audit = live_robust.build_audit_report_from_logs(
        log_dir,
        schema_module=full_schema,
        max_files=args.audit_max_files,
        corr_threshold=args.audit_corr_threshold,
    )
    names = live_robust.select_robust_features(base_names, audit_report=audit)
    print(
        f"Feature audit: base={len(base_names)} robust={len(names)} "
        f"audit_flagged={audit.get('n_flagged', 0)} "
        f"live_chunks={audit.get('n_chunks', 0)}",
        flush=True,
    )
    audit_path = args.out_dir / "feature_audit.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return names, audit


def featurize(chunks: list[list[dict]], names: list[str]) -> np.ndarray:
    rows = []
    for i, hands in enumerate(chunks):
        feat = schema.extract_chunk_features(hands, feature_names=names)
        rows.append([float(feat.get(n, 0.0)) for n in names])
        if (i + 1) % 500 == 0:
            print(f"  featurized {i + 1}/{len(chunks)}", flush=True)
    return np.asarray(rows, dtype=np.float64)


def score_live_model_windows(
    model: CompetitiveMinerModel, chunks: list[list[dict]], y: np.ndarray, *, window: int = 100
) -> dict:
    final = np.zeros(len(chunks), dtype=float)
    for start in range(0, len(chunks), window):
        sl = slice(start, start + window)
        scores = model.score_chunks(chunks[sl])
        final[sl] = np.asarray(scores, dtype=float)
    return eval_suite(y, final, f"live_model_bs{window}")


def combined_gate_score(topk100: dict, topk120: dict) -> float:
    """Average topk reward at batch sizes 100 and 120."""
    return 0.5 * float(topk100["reward"]) + 0.5 * float(topk120["reward"])


def stratified_split_indices(y: np.ndarray, *, seed: int, cal_fraction: float = 0.5):
    """Split indices into calibration vs gate, preserving label balance."""
    rng = np.random.default_rng(seed)
    cal_idx: list[int] = []
    gate_idx: list[int] = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        cut = int(round(len(idx) * cal_fraction))
        cut = min(max(cut, 1 if len(idx) else 0), max(0, len(idx) - 1) if len(idx) > 1 else len(idx))
        cal_idx.extend(idx[:cut].tolist())
        gate_idx.extend(idx[cut:].tolist())
    if not gate_idx and cal_idx:
        gate_idx.append(cal_idx.pop())
    if not cal_idx and gate_idx:
        cal_idx.append(gate_idx.pop())
    rng.shuffle(cal_idx)
    rng.shuffle(gate_idx)
    return np.asarray(cal_idx, dtype=int), np.asarray(gate_idx, dtype=int)


def current_was_force_deployed() -> bool:
    report_path = COMPETITIVE_DIR / "train_report.json"
    if not report_path.exists():
        return False
    try:
        report = read_json(report_path)
    except Exception:
        return False
    gate = report.get("gate") or {}
    return bool(gate.get("forced"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--request-log-dir", type=Path, default=REQUEST_LOG_DIR)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--recent-val-days", type=int, default=4)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gain", type=float, default=0.002)
    ap.add_argument("--audit-max-files", type=int, default=15)
    ap.add_argument("--audit-corr-threshold", type=float, default=0.70)
    ap.add_argument(
        "--leakage-credit",
        type=float,
        default=0.05,
        help="Credit applied to candidate when live current was force-deployed (holdout leak).",
    )
    ap.add_argument(
        "--deploy-to-competitive",
        action="store_true",
        help="If gate passes, write to models/competitive/current.joblib",
    )
    ap.add_argument("--force-deploy", action="store_true")
    args = ap.parse_args()

    if not args.examples.exists():
        raise SystemExit(f"Missing examples: {args.examples}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    names, audit = resolve_feature_names(args, args.request_log_dir)
    print(f"live_geometry features = {len(names)}", flush=True)

    print(f"Loading raw chunks from {args.examples}...", flush=True)
    chunks, y, dates = load_raw_examples(args.examples)
    unique = sorted(set(dates.tolist()))
    holdout_dates = unique[-args.holdout_days :]
    recent_dates = unique[-(args.holdout_days + args.recent_val_days) : -args.holdout_days]
    hold_set = set(holdout_dates)
    pool_mask = np.array([d not in hold_set for d in dates])
    hold_mask = ~pool_mask
    print(
        f"n={len(y)} dates={unique[0]}..{unique[-1]} holdout={holdout_dates} "
        f"lodo={recent_dates} pool={pool_mask.sum()} hold={hold_mask.sum()}",
        flush=True,
    )

    # Traffic-matched aug: majority 80–100, light xlarge (see LargeAugmentationConfig defaults).
    aug_cfg = LargeAugmentationConfig(
        large_ratio=1.00,
        medium_ratio=0.30,
        small_live_ratio=0.25,
        xlarge_ratio=0.20,
    )
    pool_idx = np.flatnonzero(pool_mask)
    hold_idx = np.flatnonzero(hold_mask)
    pool_chunks = [chunks[i] for i in pool_idx]
    pool_y = y[pool_mask]
    pool_dates = dates[pool_mask]

    print("Augmenting pool (traffic-matched 80–100 heavy)...", flush=True)
    train_chunks, train_y, train_dates, aug_stats = build_training_views(
        pool_chunks, pool_y, pool_dates.tolist(), aug_cfg, seed=args.seed
    )
    print(f"  aug stats={aug_stats}", flush=True)

    hold_chunks = [chunks[i] for i in hold_idx]
    hold_y = y[hold_mask]
    hold_dates = [str(d) for d in dates[hold_mask].tolist()]

    print("Featurizing train...", flush=True)
    X_train = featurize(train_chunks, names)
    y_train = np.asarray(train_y, dtype=np.int64)
    dates_train = np.asarray(train_dates)
    train_pool_mask = np.ones(len(y_train), dtype=bool)

    print("LODO monitor (not used for calibration)...", flush=True)
    oof, oy = lodo_oof(
        X_train,
        y_train,
        dates_train,
        train_pool_mask,
        recent_dates,
        X_hn=None,
        y_hn=None,
        hn_weight=0.0,
        half_life=args.half_life_days,
        seed=args.seed,
        model_factory=make_xgb,
    )
    # LODO-only metrics for monitoring; calibrator below is live-shaped.
    _, _, m_lodo_raw = tune_cal_and_logit(oy, oof, args.seed)
    m_lodo = m_lodo_raw
    print(
        f"  LODO(monitor) reward={m_lodo['reward']:.4f} ap={m_lodo['ap']:.4f} "
        f"bot@5fpr={m_lodo['bot_recall_at_5fpr']:.4f}",
        flush=True,
    )

    w_pool = recency_weights(dates_train, sorted(set(dates_train.tolist())), args.half_life_days)
    model_pool = make_xgb(args.seed)
    model_pool.fit(X_train, y_train, sample_weight=w_pool)

    print("Building live-distribution holdout (70–160 hands)...", flush=True)
    live_chunks, live_y, _ = build_live_distribution_holdout(
        hold_chunks, hold_y, hold_dates, seed=args.seed + 7, ratio=1.0
    )
    hand_counts = [len(c) for c in live_chunks]
    print(
        f"  live-shaped n={len(live_y)} hand_count min/med/max="
        f"{min(hand_counts)}/{int(np.median(hand_counts))}/{max(hand_counts)}",
        flush=True,
    )

    print("Featurizing live holdout...", flush=True)
    X_live = featurize(live_chunks, names)
    raw_live = model_pool.predict_proba(X_live)[:, 1]

    cal_idx, gate_idx = stratified_split_indices(live_y, seed=args.seed + 11, cal_fraction=0.5)
    print(
        f"Live split: cal={len(cal_idx)} gate={len(gate_idx)} "
        f"(cal for tune_cal_and_logit, gate for deploy decision)",
        flush=True,
    )

    print("Calibrating on live-shaped cal slice...", flush=True)
    cal, logit, m_cal = tune_cal_and_logit(live_y[cal_idx], raw_live[cal_idx], args.seed)
    print(
        f"  live-cal reward={m_cal['reward']:.4f} logit={logit}",
        flush=True,
    )

    gate_chunks = [live_chunks[i] for i in gate_idx]
    gate_y = live_y[gate_idx]
    cand_scores = apply_post(raw_live[gate_idx], calibrator=cal, logit=logit)
    cand_live = eval_suite(gate_y, cand_scores, "candidate_live_shaped_gate")
    cand_topk100 = simulate_batch_reward(gate_y, cand_scores, batch_size=100, fraction=0.125)
    cand_topk120 = simulate_batch_reward(gate_y, cand_scores, batch_size=120, fraction=0.125)
    cand_gate = combined_gate_score(cand_topk100, cand_topk120)
    print(
        f"  CANDIDATE live reward={cand_live['reward']:.4f} "
        f"topk100={cand_topk100['reward']:.4f} topk120={cand_topk120['reward']:.4f} "
        f"combined_gate={cand_gate:.4f} ap={cand_live['ap']:.4f}",
        flush=True,
    )

    current_path = COMPETITIVE_DIR / "current.joblib"
    current_metric = None
    current_topk100 = None
    current_topk120 = None
    current_gate = None
    current_version = None
    if current_path.exists():
        print("Scoring CURRENT live model on same gate slice...", flush=True)
        live_model = CompetitiveMinerModel(model_path=current_path)
        current_version = live_model.model_version
        current_topk100 = score_live_model_windows(live_model, gate_chunks, gate_y, window=100)
        current_topk120 = score_live_model_windows(live_model, gate_chunks, gate_y, window=120)
        current_gate = combined_gate_score(current_topk100, current_topk120)
        current_metric = current_topk100
        print(
            f"  CURRENT ({current_version}) topk100={current_topk100['reward']:.4f} "
            f"topk120={current_topk120['reward']:.4f} combined_gate={current_gate:.4f}",
            flush=True,
        )
    else:
        print("  No current.joblib; candidate deploys if --deploy-to-competitive.", flush=True)

    cur_gate_val = current_gate if current_gate is not None else -1.0
    forced_current = current_was_force_deployed()
    leakage_credit = float(args.leakage_credit) if forced_current else 0.0
    adjusted_cand = cand_gate + leakage_credit
    gain = cand_gate - cur_gate_val
    adjusted_gain = adjusted_cand - cur_gate_val
    beat = (
        (current_gate is None)
        or (adjusted_gain >= args.min_gain)
        or args.force_deploy
    )

    print(
        f"\nGATE: candidate combined={cand_gate:.4f} vs current={cur_gate_val:.4f} "
        f"gain={gain:+.4f} leakage_credit={leakage_credit:.4f} "
        f"(forced_current={forced_current}) adjusted_gain={adjusted_gain:+.4f} "
        f"min_gain={args.min_gain:.4f} -> {'DEPLOY' if beat else 'HOLD'}",
        flush=True,
    )
    print(
        "Deploy artifact = pool-held-out model + live-shaped calibrator "
        "(no all-data holdout refit).",
        flush=True,
    )

    trained_at = datetime.now(timezone.utc).isoformat()
    fingerprint = recipe_fingerprint(ROOT)
    report = {
        "trained_at_utc": trained_at,
        "model_name": "poker44-beat-v3-coherent-live",
        "model_version": MODEL_VERSION,
        "n_features": len(names),
        "feature_set": "beat_v3_coherent_live",
        "recipe_fingerprint": fingerprint,
        "half_life_days": args.half_life_days,
        "score_remap": logit,
        "holdout_dates": holdout_dates,
        "recent_lodo_dates": recent_dates,
        "latest_source_date": unique[-1],
        "benchmark_n": int(len(y)),
        "feature_audit_summary": {
            "n_base": len(full_schema.FEATURE_NAMES),
            "n_robust": len(names),
            "n_audit_flagged": audit.get("n_flagged", 0),
            "live_hand_count_median": audit.get("hand_count_median"),
        },
        "augmentation": {
            "config": aug_cfg.as_dict(),
            "pool_stats": aug_stats,
            "deploy_stats": aug_stats,
            "note": "deploy uses same pool-aug train set (no holdout refit)",
        },
        "calibration": {
            "source": "live_shaped_cal_slice",
            "n_cal": int(len(cal_idx)),
            "n_gate": int(len(gate_idx)),
            "live_cal_metrics": m_cal,
            "logit": logit,
        },
        "gate": {
            "candidate_combined_topk": cand_gate,
            "candidate_topk100": cand_topk100["reward"],
            "candidate_topk120": cand_topk120["reward"],
            "current_combined_topk": cur_gate_val,
            "current_topk100": current_topk100["reward"] if current_topk100 else None,
            "current_topk120": current_topk120["reward"] if current_topk120 else None,
            "current_version": current_version,
            "gain": gain,
            "adjusted_gain": adjusted_gain,
            "leakage_credit": leakage_credit,
            "forced_current_baseline": forced_current,
            "min_gain": args.min_gain,
            "deployed": False,
            "forced": bool(args.force_deploy),
            "fair_deploy": True,
        },
        "metrics": {
            "lodo_monitor": m_lodo,
            "lodo_selection_score": selection_score(m_lodo),
            "candidate_live_shaped": cand_live,
            "candidate_topk100": cand_topk100,
            "candidate_topk120": cand_topk120,
            "current_live_shaped": current_metric,
        },
        "selection_policy": (
            "Traffic-matched aug + live-robust OOD drops; calibrate on live-shaped "
            "cal slice; gate on disjoint live-shaped slice; deploy the gated pool "
            "model only (no holdout-date refit). Optional leakage credit when live "
            "current was force-deployed."
        ),
    }

    metadata = {
        "model_name": report["model_name"],
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
        "feature_set": "beat_v3_coherent_live",
        "holdout_dates": holdout_dates,
        "latest_source_date": unique[-1],
        "trained_at_utc": trained_at,
        "framework": "xgb_beat_v3_coherent_live",
        "recipe_fingerprint": fingerprint,
        "fair_deploy": True,
    }
    artifact = {
        "kind": "single",
        "models": [model_pool],
        "weights": [1.0],
        "calibrator": cal,
        "feature_names": names,
        "feature_set": "beat_v3_coherent_live",
        "metadata": metadata,
    }

    staging_artifact = args.out_dir / "candidate_live_geometry.joblib"
    atomic_joblib_dump(artifact, staging_artifact)
    atomic_write_text(args.out_dir / "estimation_report.json", json.dumps(report, indent=2) + "\n")
    print(f"Staging artifact -> {staging_artifact}", flush=True)

    deploy_path = COMPETITIVE_DIR / "current.joblib"
    should_deploy = beat and args.deploy_to_competitive
    if should_deploy:
        report["gate"]["deployed"] = True
        if deploy_path.exists():
            arch = COMPETITIVE_DIR / "archive"
            arch.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            shutil.copy2(deploy_path, arch / f"current_{stamp}.joblib")
            prev_report = COMPETITIVE_DIR / "train_report.json"
            if prev_report.exists():
                shutil.copy2(prev_report, arch / f"train_report_{stamp}.json")
            prune_archive(arch, keep=int(os.getenv("POKER44_ARCHIVE_KEEP", "30")))
        atomic_joblib_dump(artifact, deploy_path)
        atomic_write_text(COMPETITIVE_DIR / "train_report.json", json.dumps(report, indent=2) + "\n")
        atomic_write_text(
            COMPETITIVE_DIR / "threshold.json",
            json.dumps(
                {
                    "threshold": 0.5,
                    "batch_calibration": "topk_v1",
                    "score_remap": logit,
                    "model_name": report["model_name"],
                    "model_version": MODEL_VERSION,
                    "latest_source_date": unique[-1],
                    "recipe_fingerprint": fingerprint,
                    "model_path": str(deploy_path),
                    "fair_deploy": True,
                },
                indent=2,
            )
            + "\n",
        )
        print(f"DEPLOYED -> {deploy_path}", flush=True)
    elif beat:
        print("Gate PASS but --deploy-to-competitive not set; staging only.", flush=True)
    else:
        print("HELD: candidate did not beat current on combined topk gate.", flush=True)

    print(json.dumps({"gate": report["gate"]}, indent=2))
    # Always 0 when training/gate completed; HOLD is a valid outcome (not a crash).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
