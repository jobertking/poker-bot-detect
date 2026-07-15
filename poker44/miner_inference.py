"""Competitive miner inference: single or blend heads + cal + logit + top-K."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from features import (
    beat_v3_schema,
    chunk_features,
    competitive_fn_schema,
    competitive_schema,
    merged_schema,
    selective_schema,
)
from poker44.batch_calibration import (
    apply_batch_safety_topk_v1,
    apply_clip_below,
    apply_threshold_logit_v1,
    clamp01,
)
from poker44.validator.payload_view import prepare_hand_for_miner

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "competitive" / "current.joblib"

_SCHEMAS = {
    "competitive": competitive_schema,
    "competitive_fn": competitive_fn_schema,
    "beat_v3": beat_v3_schema,
    "merged": merged_schema,
    "selective": selective_schema,
    "v3": chunk_features,
}


def sanitize_chunk(hands: Sequence[dict] | None) -> list[dict]:
    out = []
    for hand in hands or []:
        if isinstance(hand, dict):
            out.append(prepare_hand_for_miner(hand))
    return out


def _blend_predict(models: list, weights: list[float], X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.asarray([], dtype=np.float64)
    blend = np.zeros(X.shape[0], dtype=np.float64)
    wsum = 0.0
    for model, weight in zip(models, weights):
        blend += float(weight) * model.predict_proba(X)[:, 1]
        wsum += float(weight)
    if wsum > 0:
        blend /= wsum
    return blend


class CompetitiveMinerModel:
    """Load joblib: single-head or blend_v1 multi-head artifact."""

    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path or os.getenv("POKER44_MODEL_PATH") or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Missing competitive model: {path}")
        payload = joblib.load(path)
        self.path = path
        self.metadata = dict(payload.get("metadata") or {})
        self.calibrator = payload.get("calibrator")
        self.threshold = float(self.metadata.get("decision_threshold", 0.5))
        self.score_remap = dict(self.metadata.get("score_remap") or payload.get("score_remap") or {})
        self.batch_mode = os.getenv(
            "POKER44_BATCH_CALIBRATION",
            str(self.metadata.get("batch_calibration_default", "topk_v1")),
        )
        self.model_name = str(self.metadata.get("model_name", "poker44-competitive-v2"))
        self.model_version = str(self.metadata.get("model_version", "5.1.0"))
        self.kind = str(payload.get("kind") or self.metadata.get("artifact_kind") or "single")

        if self.kind == "blend_v1" or "heads" in payload:
            self.heads = list(payload["heads"])
            # For manifest / logging convenience
            self.feature_names = list(self.heads[0].get("feature_names") or [])
            self.models = []
            self.weights = []
            self.feature_set = "blend"
        else:
            self.heads = [
                {
                    "name": "primary",
                    "feature_set": str(
                        payload.get("feature_set")
                        or self.metadata.get("feature_set")
                        or "competitive"
                    ),
                    "feature_names": list(
                        payload.get("feature_names")
                        or competitive_schema.FEATURE_NAMES
                    ),
                    "models": payload["models"],
                    "weights": list(
                        payload.get("weights")
                        or [1.0 / max(1, len(payload["models"]))] * len(payload["models"])
                    ),
                    "blend_weight": 1.0,
                }
            ]
            self.feature_names = self.heads[0]["feature_names"]
            self.models = self.heads[0]["models"]
            self.weights = self.heads[0]["weights"]
            self.feature_set = self.heads[0]["feature_set"]

    def _vectorize_head(self, head: dict[str, Any], chunks: Sequence[Sequence[dict] | None]) -> np.ndarray:
        schema = _SCHEMAS.get(head["feature_set"], competitive_schema)
        extract = schema.extract_chunk_features
        names = list(head["feature_names"])
        rows = []
        for chunk in chunks:
            sanitized = sanitize_chunk(chunk)
            try:
                feat = extract(sanitized, feature_names=names)
            except TypeError:
                feat = extract(sanitized)
            rows.append([float(feat.get(name, 0.0)) for name in names])
        return np.asarray(rows, dtype=np.float64)

    def _raw_scores(self, chunks: Sequence[Sequence[dict] | None]) -> np.ndarray:
        if not chunks:
            return np.asarray([], dtype=np.float64)
        total = np.zeros(len(chunks), dtype=np.float64)
        wsum = 0.0
        for head in self.heads:
            bw = float(head.get("blend_weight", 1.0))
            if bw <= 0:
                continue
            X = self._vectorize_head(head, chunks)
            total += bw * _blend_predict(head["models"], list(head["weights"]), X)
            wsum += bw
        if wsum > 0:
            total /= wsum
        return total

    def _calibrate(self, scores: np.ndarray) -> np.ndarray:
        if self.calibrator is None or scores.size == 0:
            return scores
        return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

    def _apply_score_remap(self, scores: list[float]) -> list[float]:
        if not scores or not self.score_remap:
            return [clamp01(s) for s in scores]
        if self.score_remap.get("kind") != "threshold_logit_v1":
            return [clamp01(s) for s in scores]
        return apply_threshold_logit_v1(
            scores,
            threshold=float(self.score_remap.get("threshold", 0.46)),
            temperature=float(self.score_remap.get("temperature", 0.08)),
        )

    def _batch_postprocess(self, scores: list[float]) -> list[float]:
        mode = (self.batch_mode or "topk_v1").lower()
        if mode in ("none", "off", "raw"):
            return [round(clamp01(s), 6) for s in scores]
        if mode == "clip_below":
            return apply_clip_below(scores)
        cfg = self.metadata.get("batch_safety_budget") or {}
        return apply_batch_safety_topk_v1(
            scores,
            max_positive_fraction=float(cfg.get("max_positive_fraction", 0.125)),
            max_positive_count=int(cfg.get("max_positive_count", 40)),
            positive_floor=float(cfg.get("positive_floor", 0.501)),
            positive_ceiling=float(cfg.get("positive_ceiling", 0.509)),
            negative_ceiling=float(cfg.get("negative_ceiling", 0.49)),
        )

    def score_chunks(self, chunks: Sequence[Sequence[dict] | None]) -> list[float]:
        if not chunks:
            return []
        raw = self._raw_scores(chunks)
        cal = self._calibrate(raw)
        remapped = self._apply_score_remap([float(x) for x in cal])
        return self._batch_postprocess(remapped)

    def predict(self, chunks: Sequence[Sequence[dict] | None]) -> tuple[list[float], list[bool]]:
        scores = self.score_chunks(chunks)
        preds = [s >= self.threshold for s in scores]
        return scores, preds

    def score_chunk(self, hands: Sequence[dict] | None) -> float:
        scores = self.score_chunks([list(hands or [])])
        return scores[0] if scores else 0.5


class XgbBotRiskModel(CompetitiveMinerModel):
    """Alias: miner historically named XgbBotRiskModel."""

    def __init__(self, model_dir: str | Path | None = None, *, threshold: float | None = None):
        model_path = None
        if model_dir is not None:
            p = Path(model_dir)
            if p.is_dir():
                cand = p / "current.joblib"
                model_path = cand if cand.exists() else p / "model.joblib"
            else:
                model_path = p
        elif os.getenv("POKER44_MODEL_DIR"):
            d = Path(os.environ["POKER44_MODEL_DIR"])
            model_path = d / "current.joblib" if d.is_dir() else d
        super().__init__(model_path)
        if threshold is not None:
            self.threshold = float(threshold)

    @property
    def model_dir(self) -> Path:
        return self.path.parent
