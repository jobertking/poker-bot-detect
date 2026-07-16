"""beat_v3 (competitive+FN+v3) union + hand-count-invariant coherent block.

Same base contract as beat_v3, plus cross-hand coherence features that stay
stable at live chunk sizes (~80-100 hands). Used identically by training and by
the miner at serving time.
"""

from __future__ import annotations

from features import beat_v3_schema as base
from features import coherent_features as coh

FEATURE_NAMES: list[str] = list(base.FEATURE_NAMES) + [
    name for name in coh.COHERENT_FEATURE_NAMES if name not in base.FEATURE_NAMES
]


def extract_chunk_features(
    hands: list[dict] | None, feature_names: list[str] | None = None
) -> dict[str, float]:
    feat = dict(base.extract_chunk_features(hands))
    feat.update(coh.coherent_feature_dict(hands))
    names = feature_names or FEATURE_NAMES
    return {name: float(feat.get(name, 0.0)) for name in names}


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]
