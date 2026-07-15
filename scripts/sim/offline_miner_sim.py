#!/usr/bin/env python3
"""Offline miner simulator: XGBoost scoring without wallet/axon/registration.

Simulates DetectionSynapse-style requests:
  chunks -> risk_scores (one float per chunk-group)
and compares against the reference heuristic + true labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.chunk_features import FEATURE_NAMES, extract_chunk_features, features_to_vector
from poker44.score.scoring import reward

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_MODEL = ROOT / "models" / "xgb_v1" / "model.json"
DEFAULT_OUT = ROOT / "models" / "xgb_v1" / "offline_sim_report.json"


def _heuristic_score_hand(hand: dict) -> float:
    """Mirror neurons/miner.py heuristic without importing bittensor."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}
    action_counts = Counter(action.get("action_type") for action in actions)
    meaningful_actions = max(
        1,
        sum(action_counts.get(kind, 0) for kind in ("call", "check", "bet", "raise", "fold")),
    )
    call_ratio = action_counts.get("call", 0) / meaningful_actions
    check_ratio = action_counts.get("check", 0) / meaningful_actions
    fold_ratio = action_counts.get("fold", 0) / meaningful_actions
    raise_ratio = action_counts.get("raise", 0) / meaningful_actions
    street_depth = len(streets) / 3.0
    showdown_flag = 1.0 if outcome.get("showdown") else 0.0
    player_count_signal = 0.0
    if players:
        player_count_signal = (6 - min(len(players), 6)) / 4.0

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    score = 0.0
    score += 0.32 * street_depth
    score += 0.22 * showdown_flag
    score += 0.18 * clamp01(call_ratio / 0.35)
    score += 0.12 * clamp01(check_ratio / 0.30)
    score += 0.08 * clamp01(player_count_signal)
    score -= 0.18 * clamp01(fold_ratio / 0.55)
    score -= 0.10 * clamp01(raise_ratio / 0.20)
    return clamp01(score)


def heuristic_score_chunk(chunk: list[dict]) -> float:
    if not chunk:
        return 0.5
    return round(sum(_heuristic_score_hand(h) for h in chunk) / len(chunk), 6)


class XgbScorer:
    def __init__(self, model_path: Path):
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        self.feature_names = list(FEATURE_NAMES)

    def score_chunk(self, hands: list[dict]) -> float:
        feat = extract_chunk_features(hands)
        vec = np.asarray([features_to_vector(feat)], dtype=np.float64)
        proba = float(self.model.predict_proba(vec)[0, 1])
        return max(0.0, min(1.0, proba))

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        if not chunks:
            return []
        rows = [features_to_vector(extract_chunk_features(c)) for c in chunks]
        X = np.asarray(rows, dtype=np.float64)
        proba = self.model.predict_proba(X)[:, 1]
        return [max(0.0, min(1.0, float(p))) for p in proba]


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rew, detail = reward(y_score, y_true)
    return {
        "n": int(y_true.size),
        "ap": float(average_precision_score(y_true, y_score)) if np.any(y_true == 1) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else None,
        "reward": float(rew),
        "bot_recall_at_5fpr": float(detail["bot_recall"]),
        "hard_bot_recall@0.5": float(detail["hard_bot_recall"]),
        "hard_fpr@0.5": float(detail["hard_fpr"]),
        "threshold_sanity_quality": float(detail["threshold_sanity_quality"]),
    }


def _load_examples(path: Path, holdout_dates: set[str] | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            if holdout_dates is not None and ex.get("sourceDate") not in holdout_dates:
                continue
            rows.append(ex)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--holdout-days", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32, help="Chunks per fake synapse request")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # Infer holdout dates = last N unique dates in examples
    all_dates = []
    with args.examples.open(encoding="utf-8") as fh:
        for line in fh:
            all_dates.append(json.loads(line)["sourceDate"])
    unique_dates = sorted(set(all_dates))
    holdout_dates = set(unique_dates[-args.holdout_days :])

    examples = _load_examples(args.examples, holdout_dates)
    if not examples:
        raise SystemExit("No examples selected for simulation")

    scorer = XgbScorer(args.model)
    y_true = np.asarray([int(ex["label"]) for ex in examples], dtype=int)
    chunks = [ex["hands"] for ex in examples]

    # --- A) Direct scoring latency ---
    t0 = time.perf_counter()
    xgb_scores = scorer.score_chunks(chunks)
    t1 = time.perf_counter()
    xgb_latency_ms = (t1 - t0) * 1000.0

    # Heuristic baseline
    t2 = time.perf_counter()
    heur_scores = [heuristic_score_chunk(c) for c in chunks]
    t3 = time.perf_counter()
    heur_latency_ms = (t3 - t2) * 1000.0

    # Contract checks
    assert len(xgb_scores) == len(chunks)
    assert all(0.0 <= s <= 1.0 for s in xgb_scores)

    # --- B) Fake DetectionSynapse batches ---
    batch_latencies = []
    batch_size = max(1, args.batch_size)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        tb0 = time.perf_counter()
        scores = scorer.score_chunks(batch)
        tb1 = time.perf_counter()
        batch_latencies.append((tb1 - tb0) * 1000.0)
        # synapse-shaped payload
        synapse = {
            "chunks": batch,
            "risk_scores": scores,
            "predictions": [s >= 0.5 for s in scores],
            "model_manifest": {
                "model_name": "poker44-xgb-v1",
                "model_version": "1.0.0",
                "framework": "xgboost",
            },
        }
        assert len(synapse["risk_scores"]) == len(synapse["chunks"])

    report = {
        "mode": "offline_miner_sim",
        "holdout_dates": sorted(holdout_dates),
        "n_examples": len(examples),
        "label_counts": dict(Counter(y_true.tolist())),
        "contract": {
            "risk_scores_len_ok": len(xgb_scores) == len(chunks),
            "scores_in_0_1": True,
        },
        "latency_ms": {
            "xgb_total": xgb_latency_ms,
            "xgb_per_chunk": xgb_latency_ms / len(chunks),
            "heuristic_total": heur_latency_ms,
            "heuristic_per_chunk": heur_latency_ms / len(chunks),
            "batch_size": batch_size,
            "batch_p50": float(np.percentile(batch_latencies, 50)),
            "batch_p95": float(np.percentile(batch_latencies, 95)),
            "n_batches": len(batch_latencies),
        },
        "metrics": {
            "xgboost": _metrics(y_true, np.asarray(xgb_scores)),
            "heuristic": _metrics(y_true, np.asarray(heur_scores)),
        },
        "delta_reward_xgb_minus_heuristic": float(
            _metrics(y_true, np.asarray(xgb_scores))["reward"]
            - _metrics(y_true, np.asarray(heur_scores))["reward"]
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
