"""XGBoost inference for Poker44 miner (runtime scoring only).

Train offline; this module only loads artifacts and predicts bot-risk scores.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from xgboost import XGBClassifier

from features.chunk_features import FEATURE_NAMES, extract_chunk_features, features_to_vector

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "xgb_v3_holdout"


class XgbBotRiskModel:
    """Load trained XGBoost and score chunk-groups for DetectionSynapse."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        threshold: float | None = None,
    ):
        model_dir = Path(
            model_dir
            or os.getenv("POKER44_MODEL_DIR")
            or DEFAULT_MODEL_DIR
        )
        model_path = model_dir / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model file: {model_path}. "
                "Train first or set POKER44_MODEL_DIR."
            )

        thr_path = model_dir / "threshold.json"
        if threshold is not None:
            self.threshold = float(threshold)
        elif thr_path.exists():
            self.threshold = float(json.loads(thr_path.read_text()).get("threshold", 0.5))
        else:
            self.threshold = float(os.getenv("POKER44_DECISION_THRESHOLD", "0.5"))

        self.model_dir = model_dir
        self.model_path = model_path
        self.feature_names = list(FEATURE_NAMES)
        self.model = XGBClassifier()
        self.model.load_model(model_path)

        meta_path = model_dir / "threshold.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.model_name = str(meta.get("model_name") or "poker44-xgb-v3-feat")
        self.model_version = str(meta.get("model_version") or "3.0.0-features")

    def score_chunk(self, hands: Sequence[dict] | None) -> float:
        scores = self.score_chunks([list(hands or [])])
        return scores[0]

    def score_chunks(self, chunks: Sequence[Sequence[dict] | None]) -> list[float]:
        if not chunks:
            return []
        rows = []
        for chunk in chunks:
            feat = extract_chunk_features(list(chunk or []))
            rows.append(features_to_vector(feat))
        X = np.asarray(rows, dtype=np.float64)
        # Guard feature width mismatches
        if X.shape[1] != len(self.feature_names):
            raise RuntimeError(
                f"Feature width mismatch: got {X.shape[1]} expected {len(self.feature_names)}"
            )
        proba = self.model.predict_proba(X)[:, 1]
        return [float(max(0.0, min(1.0, p))) for p in proba]

    def predict(self, chunks: Sequence[Sequence[dict] | None]) -> tuple[list[float], list[bool]]:
        scores = self.score_chunks(chunks)
        preds = [s >= self.threshold for s in scores]
        return scores, preds
