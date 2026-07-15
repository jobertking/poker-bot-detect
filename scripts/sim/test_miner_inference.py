#!/usr/bin/env python3
"""Smoke-test miner inference path without wallet/axon/bittensor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker44.miner_inference import XgbBotRiskModel


def main() -> int:
    model = XgbBotRiskModel()
    examples = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
    chunks = []
    labels = []
    with examples.open() as fh:
        for i, line in enumerate(fh):
            if i >= 16:
                break
            ex = json.loads(line)
            chunks.append(ex["hands"])
            labels.append(int(ex["label"]))

    # Mimic validator request payload: list of chunk-groups
    request = {"chunks": chunks}
    scores, preds = model.predict(request["chunks"])
    response = {
        "risk_scores": scores,
        "predictions": preds,
        "model_name": model.model_name,
        "model_version": model.model_version,
    }

    assert len(response["risk_scores"]) == len(request["chunks"])
    assert all(0.0 <= s <= 1.0 for s in response["risk_scores"])

    correct = sum(int(p) == y for p, y in zip(preds, labels))
    print(
        json.dumps(
            {
                "ok": True,
                "is_inference": True,
                "meaning": "Train offline; miner only predicts risk_scores for validator chunks.",
                "model_dir": str(model.model_dir),
                "threshold": model.threshold,
                "n_chunks": len(chunks),
                "sample_accuracy_at_threshold": round(correct / len(labels), 4),
                "mean_score": round(sum(scores) / len(scores), 4),
                "response_keys": list(response.keys()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
