"""Competitive schema + FN stealth patches."""

from __future__ import annotations

from features import competitive_schema as comp
from features import fn_patches

FEATURE_NAMES: list[str] = list(comp.FEATURE_NAMES) + list(fn_patches.FEATURE_NAMES)


def extract_chunk_features(hands: list[dict] | None) -> dict[str, float]:
    feat = dict(comp.extract_chunk_features(hands))
    feat.update(fn_patches.extract_fn_patch_features(hands))
    return {name: float(feat.get(name, 0.0)) for name in FEATURE_NAMES}


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]
