"""Competitive + FN patches + v3 cues (union) to reclaim xgb_v3 ranking headroom."""

from __future__ import annotations

from features import chunk_features as v3
from features import competitive_fn_schema as cfn

FEATURE_NAMES: list[str] = list(cfn.FEATURE_NAMES) + [
    name for name in v3.FEATURE_NAMES if name not in cfn.FEATURE_NAMES
]


def extract_chunk_features(hands: list[dict] | None) -> dict[str, float]:
    feat = dict(cfn.extract_chunk_features(hands))
    feat.update(v3.extract_chunk_features(hands))
    return {name: float(feat.get(name, 0.0)) for name in FEATURE_NAMES}


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]
