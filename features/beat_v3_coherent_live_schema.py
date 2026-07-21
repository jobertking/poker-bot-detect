"""beat_v3_coherent with live-robust feature subset (hand-count invariant)."""

from __future__ import annotations

from features import beat_v3_coherent_schema as base
from features import live_robust

FEATURE_NAMES: list[str] = live_robust.filter_feature_names(base.FEATURE_NAMES)


def extract_chunk_features(
    hands: list[dict] | None, feature_names: list[str] | None = None
) -> dict[str, float]:
    feat = base.extract_chunk_features(hands)
    names = feature_names or FEATURE_NAMES
    return {name: float(feat.get(name, 0.0)) for name in names}


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]
