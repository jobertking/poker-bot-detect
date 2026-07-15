"""Merged competitive (winner) + v3 chunk cues for ranking headroom."""

from __future__ import annotations

from features import chunk_features as v3
from features import competitive_schema as comp

# Competitive names first; add v3 names that aren't already present.
FEATURE_NAMES: list[str] = list(comp.FEATURE_NAMES) + [
    name for name in v3.FEATURE_NAMES if name not in comp.FEATURE_NAMES
]


def extract_chunk_features(hands: list[dict] | None) -> dict[str, float]:
    feat = dict(comp.extract_chunk_features(hands))
    feat.update(v3.extract_chunk_features(hands))
    return {name: float(feat.get(name, 0.0)) for name in FEATURE_NAMES}


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]
