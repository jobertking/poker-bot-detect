"""Competitive miner inference: single or blend heads + cal + logit + top-K."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from features import (
    beat_v3_coherent_live_schema,
    beat_v3_coherent_schema,
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
    "beat_v3_coherent": beat_v3_coherent_schema,
    "beat_v3_coherent_live": beat_v3_coherent_live_schema,
    "merged": merged_schema,
    "selective": selective_schema,
    "v3": chunk_features,
}


def sanitize_chunk(hands: Sequence[dict] | None) -> list[dict]:
    """Prepare hands for features. Idempotent if already miner-canonical."""
    out = []
    for hand in hands or []:
        if isinstance(hand, dict):
            out.append(prepare_hand_for_miner(hand))
    return out


def _default_feature_names(feature_set: str) -> list[str]:
    schema = _SCHEMAS.get(feature_set, competitive_schema)
    return list(getattr(schema, "FEATURE_NAMES", competitive_schema.FEATURE_NAMES))


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


def within_batch_percentile(values: np.ndarray) -> np.ndarray:
    """Average-rank percentiles in (0,1) within one request. Tie-safe, no scipy."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return arr
    if n == 1:
        return np.array([0.5], dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    sorted_v = arr[order]
    base = np.arange(1, n + 1, dtype=np.float64)
    ranks_sorted = base.copy()
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks_sorted[i : j + 1] = (base[i] + base[j]) / 2.0
        i = j + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return (ranks - 0.5) / n


def resolve_model_path(
    model_dir: str | Path | None = None,
    *,
    model_path: str | Path | None = None,
) -> Path:
    """Resolve artifact path. Precedence: explicit path → MODEL_PATH → model_dir/MODEL_DIR → default."""
    if model_path is not None:
        return Path(model_path)
    env_path = os.getenv("POKER44_MODEL_PATH")
    if env_path:
        return Path(env_path)
    if model_dir is not None:
        p = Path(model_dir)
        if p.is_dir():
            cand = p / "current.joblib"
            return cand if cand.exists() else p / "model.joblib"
        return p
    env_dir = os.getenv("POKER44_MODEL_DIR")
    if env_dir:
        d = Path(env_dir)
        if d.is_dir():
            return d / "current.joblib"
        return d
    return DEFAULT_MODEL_PATH


class CompetitiveMinerModel:
    """Load joblib: single-head or blend_v1 multi-head artifact. Hot-reloads on mtime change."""

    def __init__(self, model_path: str | Path | None = None):
        path = resolve_model_path(model_path=model_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing competitive model: {path}")
        self.path = path
        self._mtime: float | None = None
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        path = self.path
        payload = joblib.load(path)
        try:
            self._mtime = path.stat().st_mtime
        except OSError:
            self._mtime = None
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

        if self.kind == "stacked_v1" or "meta_model" in payload:
            self.kind = "stacked_v1"
            feature_set = str(
                payload.get("feature_set")
                or self.metadata.get("feature_set")
                or "beat_v3_coherent"
            )
            self.feature_set = feature_set
            self.feature_names = list(
                payload.get("feature_names") or _default_feature_names(feature_set)
            )
            self.tabular_models = list(
                payload.get("tabular_models") or payload.get("models") or []
            )
            self.tabular_weights = list(
                payload.get("tabular_weights")
                or [1.0 / max(1, len(self.tabular_models))] * len(self.tabular_models)
            )
            self.sequence_model = payload.get("sequence_model")
            self.meta_model = payload.get("meta_model")
            self.meta_inputs = list(payload.get("meta_inputs") or ["tabular", "sequence"])
            self.heads = []
            self.models = []
            self.weights = []
        elif self.kind == "rank_blend_v1" or "branches" in payload:
            self.kind = "rank_blend_v1"
            self.branches = list(payload["branches"])
            feature_set = str(
                payload.get("feature_set")
                or self.metadata.get("feature_set")
                or "beat_v3_coherent"
            )
            self.feature_set = feature_set
            self.feature_names = list(
                payload.get("feature_names")
                or self.branches[0].get("feature_names")
                or _default_feature_names(feature_set)
            )
            self.min_rank_batch = int(
                payload.get("min_rank_batch", self.metadata.get("min_rank_batch", 8))
            )
            self.heads = []
            self.models = []
            self.weights = []
        elif self.kind == "blend_v1" or "heads" in payload:
            self.heads = list(payload["heads"])
            self.feature_names = list(self.heads[0].get("feature_names") or [])
            self.models = []
            self.weights = []
            self.feature_set = "blend"
        else:
            feature_set = str(
                payload.get("feature_set")
                or self.metadata.get("feature_set")
                or "competitive"
            )
            self.heads = [
                {
                    "name": "primary",
                    "feature_set": feature_set,
                    "feature_names": list(
                        payload.get("feature_names") or _default_feature_names(feature_set)
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
            self.feature_set = feature_set

    def maybe_reload(self) -> bool:
        """Reload artifact if file mtime changed. Returns True when reloaded."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        if self._mtime is not None and mtime == self._mtime:
            return False
        self._load_from_disk()
        return True

    def _vectorize(
        self,
        feature_set: str,
        feature_names: Sequence[str],
        chunks: Sequence[Sequence[dict] | None],
    ) -> np.ndarray:
        schema = _SCHEMAS.get(feature_set, competitive_schema)
        extract = schema.extract_chunk_features
        names = list(feature_names)
        rows = []
        for chunk in chunks:
            sanitized = sanitize_chunk(chunk)
            try:
                feat = extract(sanitized, feature_names=names)
            except TypeError:
                feat = extract(sanitized)
            rows.append([float(feat.get(name, 0.0)) for name in names])
        return np.asarray(rows, dtype=np.float64)

    def _vectorize_head(self, head: dict[str, Any], chunks: Sequence[Sequence[dict] | None]) -> np.ndarray:
        return self._vectorize(head["feature_set"], head["feature_names"], chunks)

    def _rank_blend_scores(self, chunks: Sequence[Sequence[dict] | None]) -> np.ndarray:
        X = self._vectorize(self.feature_set, self.feature_names, chunks)
        n = X.shape[0]
        if n == 0:
            return np.asarray([], dtype=np.float64)
        probs = []
        weights = []
        for branch in self.branches:
            model = branch["model"]
            scaler = branch.get("scaler")
            X_in = scaler.transform(X) if scaler is not None else X
            probs.append(model.predict_proba(X_in)[:, 1])
            weights.append(float(branch.get("weight", 1.0)))
        weights_arr = np.asarray(weights, dtype=np.float64)
        wsum = float(weights_arr.sum()) or 1.0
        if n >= self.min_rank_batch:
            # Equalize branch scales via within-request percentiles before fusing.
            matrix = np.vstack([within_batch_percentile(p) for p in probs])
        else:
            matrix = np.vstack(probs)
        return (weights_arr[:, None] * matrix).sum(axis=0) / wsum

    def _stacked_raw_scores(self, chunks: Sequence[Sequence[dict] | None]) -> np.ndarray:
        """Stacked meta-probability from tabular base(s) + sequence model."""
        n = len(chunks)
        if n == 0:
            return np.asarray([], dtype=np.float64)
        sanitized = [sanitize_chunk(c) for c in chunks]
        tab = _blend_predict(
            self.tabular_models, self.tabular_weights,
            self._vectorize(self.feature_set, self.feature_names, sanitized),
        )
        seq = np.asarray(
            self.sequence_model.predict_proba(sanitized)[:, 1], dtype=np.float64
        )
        by_name = {"tabular": tab, "sequence": seq}
        cols = [by_name[name] for name in self.meta_inputs]
        Z = np.column_stack(cols)
        return np.asarray(self.meta_model.predict_proba(Z)[:, 1], dtype=np.float64)

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
            allow = os.getenv("POKER44_ALLOW_CLIP_BELOW", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if not allow:
                # clip_below zeros threshold_sanity under current reward → refuse by default.
                mode = "topk_v1"
            else:
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
        self.maybe_reload()
        if not chunks:
            return []
        if self.kind == "rank_blend_v1":
            # Fused rank scores are already in [0,1]; skip prob calibration/remap.
            raw = self._rank_blend_scores(chunks)
            return self._batch_postprocess([float(x) for x in raw])
        if self.kind == "stacked_v1":
            raw = self._stacked_raw_scores(chunks)
            cal = self._calibrate(raw)
            remapped = self._apply_score_remap([float(x) for x in cal])
            return self._batch_postprocess(remapped)
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
        # MODEL_PATH wins over MODEL_DIR when both are set (see resolve_model_path).
        super().__init__(resolve_model_path(model_dir=model_dir))
        if threshold is not None:
            self.threshold = float(threshold)

    @property
    def model_dir(self) -> Path:
        return self.path.parent
