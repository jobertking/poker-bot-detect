#!/usr/bin/env python3
"""Validate miner request/response format against the validator contract.

Does NOT need a live axon, wallet, or network.
Checks the same rules used in poker44/validator/forward.py:
  - request: chunks = List[List[hand_dict]]
  - response: risk_scores length == len(chunks), each float in [0, 1]
  - predictions optional but 1:1 with scores when present
  - model_manifest is a dict when present
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker44.miner_inference import XgbBotRiskModel


def _hand_ok(hand: Any) -> list[str]:
    errs = []
    if not isinstance(hand, dict):
        return ["hand is not a dict"]
    for key in ("metadata", "players", "streets", "actions", "outcome"):
        if key not in hand:
            errs.append(f"hand missing key '{key}'")
    return errs


def validate_request(chunks: Any) -> list[str]:
    errs = []
    if not isinstance(chunks, list):
        return ["chunks must be a list"]
    if not chunks:
        errs.append("chunks is empty (validators usually send >=1 chunk)")
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, list):
            errs.append(f"chunks[{i}] must be a list of hands")
            continue
        for j, hand in enumerate(chunk[:3]):  # spot-check first hands
            for e in _hand_ok(hand):
                errs.append(f"chunks[{i}][{j}]: {e}")
    return errs


def validate_response(
    chunks: list,
    risk_scores: Any,
    predictions: Any = None,
    model_manifest: Any = None,
) -> list[str]:
    errs = []
    if risk_scores is None:
        return ["risk_scores is None (validator discards this)"]
    if not isinstance(risk_scores, list):
        return ["risk_scores must be a list"]
    if len(risk_scores) != len(chunks):
        errs.append(
            f"len(risk_scores)={len(risk_scores)} != len(chunks)={len(chunks)} "
            "(validator discards incomplete responses)"
        )
    for i, s in enumerate(risk_scores):
        try:
            v = float(s)
        except (TypeError, ValueError):
            errs.append(f"risk_scores[{i}] not float-compatible: {s!r}")
            continue
        if not (0.0 <= v <= 1.0):
            errs.append(f"risk_scores[{i}]={v} outside [0,1]")

    if predictions is not None:
        if not isinstance(predictions, list):
            errs.append("predictions must be a list when provided")
        elif len(predictions) != len(risk_scores):
            errs.append(
                f"len(predictions)={len(predictions)} != len(risk_scores)={len(risk_scores)}"
            )
        else:
            for i, p in enumerate(predictions):
                if not isinstance(p, bool):
                    errs.append(f"predictions[{i}] must be bool, got {type(p).__name__}")

    if model_manifest is not None and not isinstance(model_manifest, dict):
        errs.append("model_manifest must be a dict when provided")

    return errs


def main() -> int:
    model = XgbBotRiskModel()
    examples = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"

    # Build a validator-like request from real benchmark chunk-groups
    chunks: list[list[dict]] = []
    with examples.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 24:
                break
            ex = json.loads(line)
            chunks.append(ex["hands"])

    # Edge cases validators may send
    cases = {
        "normal_batch": chunks,
        "single_chunk": chunks[:1],
        "empty_chunk_inside": [[]] + chunks[:2],
    }

    all_errs: list[str] = []
    reports = {}
    for name, req_chunks in cases.items():
        req_errs = validate_request(req_chunks)
        scores, preds = model.predict(req_chunks)
        manifest = {
            "model_name": model.model_name,
            "model_version": model.model_version,
            "framework": "xgboost",
            "inference_mode": "remote",
        }
        resp_errs = validate_response(req_chunks, scores, preds, manifest)
        case_errs = req_errs + resp_errs
        reports[name] = {
            "n_chunks": len(req_chunks),
            "n_scores": len(scores),
            "ok": len(case_errs) == 0,
            "errors": case_errs,
            "sample_scores": [round(s, 6) for s in scores[:3]],
            "sample_predictions": preds[:3],
        }
        all_errs.extend([f"{name}: {e}" for e in case_errs])

    # Optional: if bittensor is installed, also exercise DetectionSynapse type
    synapse_check = {"available": False}
    try:
        from poker44.validator.synapse import DetectionSynapse

        syn = DetectionSynapse(chunks=chunks[:4])
        scores, preds = model.predict(syn.chunks or [])
        syn.risk_scores = scores
        syn.predictions = preds
        syn.model_manifest = {
            "model_name": model.model_name,
            "model_version": model.model_version,
        }
        syn_errs = validate_response(
            syn.chunks or [], syn.risk_scores, syn.predictions, syn.model_manifest
        )
        synapse_check = {
            "available": True,
            "ok": len(syn_errs) == 0,
            "errors": syn_errs,
            "type": type(syn).__name__,
        }
        all_errs.extend([f"DetectionSynapse: {e}" for e in syn_errs])
    except Exception as exc:
        synapse_check = {
            "available": False,
            "note": f"Skipped live Synapse class import ({exc}). Format rules still checked above.",
        }

    summary = {
        "contract": {
            "request": "DetectionSynapse(chunks: List[List[hand_dict]])",
            "response_required": "risk_scores: List[float], len == len(chunks), values in [0,1]",
            "response_optional": "predictions: List[bool], model_manifest: dict",
        },
        "cases": reports,
        "synapse_class_check": synapse_check,
        "overall_ok": len(all_errs) == 0,
        "errors": all_errs,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not all_errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
