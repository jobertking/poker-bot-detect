#!/usr/bin/env python3
"""Train a STACKED candidate (XGB beat_v3_coherent + Set-Transformer sequence
model, fused by a logistic meta-learner) and deploy ONLY if it beats the live
model on a live-shaped holdout (80-100 hand chunks, top-K at batch size 100).

Pipeline
--------
1. Split sealed holdout (last N dates) from the training pool.
2. Augment the pool to live geometry (large/medium chunk merges).
3. Base models on pool MINUS a recent meta-slice:
     - tabular: XGBoost on beat_v3_coherent features
     - sequence: hierarchical Set-Transformer over raw action tokens
   Predict both on the meta-slice -> fit logistic meta-learner -> tune
   calibrator + threshold-logit remap on the meta-slice.
4. Write the candidate artifact and score it through the REAL serving path
   (CompetitiveMinerModel) on the shared live-shaped holdout; score the current
   live model on the identical chunks. Deploy only if gain >= --min-gain.
5. On deploy, refit both base models on ALL data (+aug), reuse the meta-learner
   + calibrator + remap, archive the old artifact, and atomically publish.
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
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import beat_v3_coherent_schema as schema
from poker44.artifact_io import (
    atomic_joblib_dump,
    atomic_write_text,
    prune_archive,
    recipe_fingerprint,
)
from poker44.large_chunk_augment import (
    LargeAugmentationConfig,
    build_live_shaped_holdout,
    build_training_views,
)
from poker44.miner_inference import CompetitiveMinerModel
from poker44.sequence_model import SequenceModelConfig, SequenceModelWrapper
from scripts.train.train_beat_v3_coherent import (
    featurize,
    load_raw_examples,
    make_xgb,
    score_live_model_windows,
)
from scripts.train.train_competitive_v3 import (
    apply_post,
    eval_suite,
    recency_weights,
    selection_score,
    tune_cal_and_logit,
)

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT = ROOT / "models" / "competitive"
MODEL_VERSION = "9.0.0"


def make_seq(args, seed: int) -> SequenceModelWrapper:
    cfg = SequenceModelConfig(
        d_model=int(args.seq_d_model),
        n_heads=4,
        n_action_layers=int(args.seq_action_layers),
        n_hand_layers=1,
        dropout=0.1,
        ff_mult=2,
        max_actions_per_hand=12,
        max_hands_per_chunk=int(args.seq_max_hands),
        schema_version=2,
    )
    return SequenceModelWrapper(
        config=cfg,
        n_epochs=int(args.seq_epochs),
        batch_size=int(args.seq_batch),
        learning_rate=float(args.seq_lr),
        weight_decay=1e-4,
        val_fraction=0.1,
        early_stopping_patience=3,
        seed=seed,
        device="cpu",
        verbose=True,
    )


def fit_seq(model: SequenceModelWrapper, chunks, y, dates, half_life):
    w = recency_weights(np.asarray(dates), sorted(set(map(str, dates))), half_life)
    model.fit(list(chunks), np.asarray(y, dtype=np.int64), sample_weight=w)
    return model


def build_stacked_artifact(xgb_model, seq_model, meta_lr, cal, logit, names,
                           trained_at, fingerprint, unique, holdout_dates):
    metadata = {
        "model_name": "poker44-stacked-settransformer",
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
        "feature_set": "beat_v3_coherent",
        "holdout_dates": holdout_dates,
        "latest_source_date": unique[-1],
        "trained_at_utc": trained_at,
        "framework": "stacked_xgb_settransformer",
        "recipe_fingerprint": fingerprint,
    }
    artifact = {
        "kind": "stacked_v1",
        "feature_set": "beat_v3_coherent",
        "feature_names": names,
        "tabular_models": [xgb_model],
        "tabular_weights": [1.0],
        "sequence_model": seq_model,
        "meta_model": meta_lr,
        "meta_inputs": ["tabular", "sequence"],
        "calibrator": cal,
        "score_remap": logit,
        "metadata": metadata,
    }
    return artifact, metadata


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--recent-val-days", type=int, default=4)
    ap.add_argument("--half-life-days", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--large-ratio", type=float, default=1.25)
    ap.add_argument("--medium-ratio", type=float, default=0.40)
    ap.add_argument("--seq-epochs", type=int, default=8)
    ap.add_argument("--seq-batch", type=int, default=32)
    ap.add_argument("--seq-lr", type=float, default=1e-3)
    ap.add_argument("--seq-d-model", type=int, default=64)
    ap.add_argument("--seq-action-layers", type=int, default=2)
    ap.add_argument("--seq-max-hands", type=int, default=80)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="Required live-shaped reward gain over current to deploy.")
    ap.add_argument("--force-deploy", action="store_true",
                    help="Deploy even if it does not beat current (NOT recommended).")
    args = ap.parse_args()

    if not args.examples.exists():
        raise SystemExit(f"Missing examples: {args.examples}")

    try:
        import torch
        torch.set_num_threads(int(args.torch_threads))
    except Exception:
        pass

    names = list(schema.FEATURE_NAMES)
    print(f"beat_v3_coherent features = {len(names)}; seq max_hands={args.seq_max_hands}", flush=True)
    print(f"Loading raw chunks from {args.examples}...", flush=True)
    chunks, y, dates = load_raw_examples(args.examples)
    unique = sorted(set(dates.tolist()))
    holdout_dates = unique[-args.holdout_days:]
    recent_dates = unique[-(args.holdout_days + args.recent_val_days):-args.holdout_days]
    hold_set = set(holdout_dates)
    recent_set = set(recent_dates)
    pool_mask = np.array([d not in hold_set for d in dates])
    hold_mask = ~pool_mask
    print(
        f"n={len(y)} dates={unique[0]}..{unique[-1]} holdout={holdout_dates} "
        f"meta_slice={recent_dates} pool={pool_mask.sum()} hold={hold_mask.sum()}",
        flush=True,
    )

    aug_cfg = LargeAugmentationConfig(
        large_ratio=float(args.large_ratio), medium_ratio=float(args.medium_ratio)
    )
    pool_idx = np.flatnonzero(pool_mask)
    hold_idx = np.flatnonzero(hold_mask)
    pool_chunks = [chunks[i] for i in pool_idx]
    pool_y = y[pool_mask]
    pool_dates = dates[pool_mask]

    print("Augmenting pool (large+medium)...", flush=True)
    train_chunks, train_y, train_dates, aug_stats = build_training_views(
        pool_chunks, pool_y, pool_dates.tolist(), aug_cfg, seed=args.seed
    )
    train_y = np.asarray(train_y, dtype=np.int64)
    train_dates = np.asarray([str(d) for d in train_dates])
    print(f"  aug stats={aug_stats} n_train={len(train_y)}", flush=True)

    # meta-slice = augmented pool rows on the recent dates (base models never see these).
    meta_mask = np.array([d in recent_set for d in train_dates])
    base_mask = ~meta_mask
    if meta_mask.sum() < 20 or base_mask.sum() < 50:
        raise SystemExit(
            f"meta-slice too small (meta={meta_mask.sum()} base={base_mask.sum()}); "
            "reduce --recent-val-days or add data."
        )
    print(f"  base rows={base_mask.sum()} meta rows={meta_mask.sum()}", flush=True)

    print("Featurizing augmented train (tabular)...", flush=True)
    X_train = featurize(train_chunks)

    base_chunks = [c for c, m in zip(train_chunks, base_mask) if m]
    base_y = train_y[base_mask]
    base_dates = train_dates[base_mask]
    meta_chunks = [c for c, m in zip(train_chunks, meta_mask) if m]
    meta_y = train_y[meta_mask]

    # --- base models on base slice ---
    print("Fitting tabular base (XGB) on base slice...", flush=True)
    w_base = recency_weights(base_dates, sorted(set(base_dates.tolist())), args.half_life_days)
    xgb_base = make_xgb(args.seed)
    xgb_base.fit(X_train[base_mask], base_y, sample_weight=w_base)

    print("Fitting sequence base (Set-Transformer) on base slice...", flush=True)
    seq_base = fit_seq(make_seq(args, args.seed), base_chunks, base_y, base_dates, args.half_life_days)

    # --- meta-learner on meta slice ---
    print("Building meta-slice base predictions...", flush=True)
    tab_meta = xgb_base.predict_proba(X_train[meta_mask])[:, 1]
    seq_meta = np.asarray(seq_base.predict_proba(meta_chunks)[:, 1], dtype=np.float64)
    Z_meta = np.column_stack([tab_meta, seq_meta])
    meta_lr = LogisticRegression(C=1.0, max_iter=2000)
    meta_lr.fit(Z_meta, meta_y)
    meta_prob = meta_lr.predict_proba(Z_meta)[:, 1]
    print(
        f"  meta coef tab={meta_lr.coef_[0][0]:+.3f} seq={meta_lr.coef_[0][1]:+.3f} "
        f"intercept={meta_lr.intercept_[0]:+.3f}",
        flush=True,
    )

    cal, logit, _ = tune_cal_and_logit(meta_y, meta_prob, args.seed)
    m_meta = eval_suite(meta_y, apply_post(meta_prob, calibrator=cal, logit=logit), "meta_slice")
    # For reference: tabular-only and sequence-only on the meta slice.
    m_tab = eval_suite(meta_y, tab_meta, "meta_tab_only")
    m_seq = eval_suite(meta_y, seq_meta, "meta_seq_only")
    print(
        f"  META reward={m_meta['reward']:.4f} ap={m_meta['ap']:.4f} "
        f"bot@5fpr={m_meta['bot_recall_at_5fpr']:.4f} | tab_ap={m_tab['ap']:.4f} "
        f"seq_ap={m_seq['ap']:.4f} logit={logit}",
        flush=True,
    )

    # --- candidate artifact (base-slice models) for the gate ---
    trained_at = datetime.now(timezone.utc).isoformat()
    fingerprint = recipe_fingerprint(ROOT)
    cand_artifact, _ = build_stacked_artifact(
        xgb_base, seq_base, meta_lr, cal, logit, names,
        trained_at, fingerprint, unique, holdout_dates,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = args.out_dir / "candidate_stacked.joblib"
    atomic_joblib_dump(cand_artifact, cand_path)

    print("Building live-shaped holdout (80-100 hands)...", flush=True)
    hold_chunks = [chunks[i] for i in hold_idx]
    hold_y = y[hold_mask]
    hold_dates = [str(d) for d in dates[hold_mask].tolist()]
    live_chunks, live_y, _ = build_live_shaped_holdout(
        hold_chunks, hold_y, hold_dates, seed=args.seed + 7, ratio=1.0
    )
    print(f"  live-shaped n={len(live_y)}", flush=True)

    print("Scoring CANDIDATE through serving path on live-shaped holdout...", flush=True)
    cand_model = CompetitiveMinerModel(model_path=cand_path)
    cand_metric = score_live_model_windows(cand_model, live_chunks, live_y, window=100)
    print(
        f"  CANDIDATE live reward={cand_metric['reward']:.4f} ap={cand_metric['ap']:.4f} "
        f"bot@5fpr={cand_metric['bot_recall_at_5fpr']:.4f}",
        flush=True,
    )

    current_path = args.out_dir / "current.joblib"
    current_metric = None
    current_version = None
    if current_path.exists():
        print("Scoring CURRENT live model on same holdout...", flush=True)
        live_model = CompetitiveMinerModel(model_path=current_path)
        current_version = live_model.model_version
        current_metric = score_live_model_windows(live_model, live_chunks, live_y, window=100)
        print(
            f"  CURRENT ({current_version}) live reward={current_metric['reward']:.4f} "
            f"ap={current_metric['ap']:.4f} bot@5fpr={current_metric['bot_recall_at_5fpr']:.4f}",
            flush=True,
        )
    else:
        print("  No current.joblib present; candidate will deploy as first model.", flush=True)

    cand_gate = cand_metric["reward"]
    cur_gate = current_metric["reward"] if current_metric else -1.0
    gain = cand_gate - cur_gate
    beat = (current_metric is None) or (gain >= args.min_gain) or args.force_deploy

    print(
        f"\nGATE: candidate live reward={cand_gate:.4f} vs current={cur_gate:.4f} "
        f"gain={gain:+.4f} min_gain={args.min_gain:.4f} -> "
        f"{'DEPLOY' if beat else 'HOLD (keep current)'}",
        flush=True,
    )

    report = {
        "trained_at_utc": trained_at,
        "model_name": "poker44-stacked-settransformer",
        "model_version": MODEL_VERSION,
        "n_features": len(names),
        "feature_set": "beat_v3_coherent",
        "recipe_fingerprint": fingerprint,
        "half_life_days": args.half_life_days,
        "score_remap": logit,
        "holdout_dates": holdout_dates,
        "meta_slice_dates": recent_dates,
        "latest_source_date": unique[-1],
        "benchmark_n": int(len(y)),
        "sequence_config": make_seq(args, args.seed).config.to_dict(),
        "meta_coef": {
            "tabular": float(meta_lr.coef_[0][0]),
            "sequence": float(meta_lr.coef_[0][1]),
            "intercept": float(meta_lr.intercept_[0]),
        },
        "augmentation": {"config": aug_cfg.as_dict(), "pool_stats": aug_stats},
        "gate": {
            "candidate_live_reward": cand_gate,
            "current_live_reward": cur_gate,
            "current_version": current_version,
            "gain": gain,
            "min_gain": args.min_gain,
            "deployed": bool(beat),
            "forced": bool(args.force_deploy),
        },
        "metrics": {
            "meta_slice": m_meta,
            "meta_slice_selection_score": selection_score(m_meta),
            "meta_tab_only": m_tab,
            "meta_seq_only": m_seq,
            "candidate_live_shaped": cand_metric,
            "current_live_shaped": current_metric,
        },
        "selection_policy": (
            "stacked XGB(beat_v3_coherent)+SetTransformer via logistic meta; deploy "
            "ONLY if candidate beats current live model on live-shaped topk100 by >= min_gain"
        ),
    }

    if not beat:
        atomic_write_text(
            args.out_dir / "candidate_report.json", json.dumps(report, indent=2) + "\n"
        )
        print(
            f"HELD: candidate did not beat current. Saved aside -> {cand_path}\n"
            f"Live model untouched.",
            flush=True,
        )
        print(json.dumps({"gate": report["gate"]}, indent=2))
        return 0

    # --- deploy: refit base models on ALL data (+aug), reuse meta+cal+logit ---
    print("\nGate passed. Fitting deploy base models on all+aug...", flush=True)
    deploy_chunks, deploy_y, deploy_dates, deploy_aug = build_training_views(
        chunks, y, dates.tolist(), aug_cfg, seed=args.seed + 99
    )
    deploy_y = np.asarray(deploy_y, dtype=np.int64)
    deploy_dates = np.asarray([str(d) for d in deploy_dates])
    X_deploy = featurize(deploy_chunks)
    w_all = recency_weights(deploy_dates, sorted(set(deploy_dates.tolist())), args.half_life_days)
    xgb_full = make_xgb(args.seed)
    xgb_full.fit(X_deploy, deploy_y, sample_weight=w_all)
    seq_full = fit_seq(make_seq(args, args.seed + 1), deploy_chunks, deploy_y, deploy_dates, args.half_life_days)

    artifact, _ = build_stacked_artifact(
        xgb_full, seq_full, meta_lr, cal, logit, names,
        trained_at, fingerprint, unique, holdout_dates,
    )
    report["augmentation"]["deploy_stats"] = deploy_aug

    if current_path.exists():
        arch = args.out_dir / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(current_path, arch / f"current_{stamp}.joblib")
        prev_report = args.out_dir / "train_report.json"
        if prev_report.exists():
            shutil.copy2(prev_report, arch / f"train_report_{stamp}.json")
        prune_archive(arch, keep=int(os.getenv("POKER44_ARCHIVE_KEEP", "30")))
        print(f"Archived previous artifact under {arch}", flush=True)

    atomic_joblib_dump(artifact, current_path)
    atomic_write_text(args.out_dir / "train_report.json", json.dumps(report, indent=2) + "\n")
    atomic_write_text(
        args.out_dir / "threshold.json",
        json.dumps(
            {
                "threshold": 0.5,
                "batch_calibration": "topk_v1",
                "score_remap": logit,
                "model_name": report["model_name"],
                "model_version": MODEL_VERSION,
                "latest_source_date": unique[-1],
                "recipe_fingerprint": fingerprint,
                "model_path": str(current_path),
            },
            indent=2,
        )
        + "\n",
    )
    try:
        cand_path.unlink()
    except OSError:
        pass
    print(f"DEPLOYED stacked candidate {MODEL_VERSION} -> {current_path}", flush=True)
    print(json.dumps({"gate": report["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
