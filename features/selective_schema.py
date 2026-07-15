"""Selective merge: competitive schema + hand-picked v3 cue groups."""

from __future__ import annotations

from features import chunk_features as v3
from features import competitive_schema as comp

# Groups that historically lift ranking at low FPR (not the full 178 dump).
V3_GROUPS: dict[str, list[str]] = {
    "bigrams": [f"chunk_bg_{k}" for k in v3.BIGRAMS],
    "amount_reg": [
        "chunk_amount_cv",
        "chunk_unique_size_rate",
        "chunk_round_size_mean",
        "chunk_p10_normalized_amount_bb",
        "chunk_p90_normalized_amount_bb",
        "chunk_mean_normalized_amount_bb",
        "chunk_std_normalized_amount_bb",
    ],
    "consistency": [
        "chunk_hand_signature_max_share",
        "chunk_hand_signature_entropy",
        "chunk_fold_ratio_std",
        "chunk_aggression_std",
        "chunk_amount_mean_std",
        "chunk_entropy_std",
        "chunk_n_actions_std",
        "chunk_street_depth_std",
    ],
    "street_end": [
        "chunk_showdown_rate",
        "chunk_ended_preflop_rate",
        "chunk_ended_river_rate",
        "chunk_flop_action_share",
        "chunk_turn_action_share",
        "chunk_river_action_share",
        "chunk_preflop_action_share",
        "chunk_action_entropy",
        "chunk_fold_ratio_mean",
        "chunk_aggression_mean",
        "chunk_n_actions",
    ],
    "hand_means_key": [
        n
        for n in v3.FEATURE_NAMES
        if n.startswith(("mean_", "std_"))
        and any(
            k in n
            for k in (
                "fold_ratio",
                "aggression",
                "street_depth",
                "round_size",
                "action_entropy",
                "button_action",
                "ip_action",
                "unique_size",
                "amount_cv",
                "seat_action_entropy",
            )
        )
    ],
}

# Default deployed selection (overwritten by train script after LODO greed).
DEFAULT_SELECTED_GROUPS: list[str] = [
    "bigrams",
    "amount_reg",
    "consistency",
]


def feature_names_for_groups(groups: list[str] | None = None) -> list[str]:
    groups = list(groups or DEFAULT_SELECTED_GROUPS)
    extra: list[str] = []
    seen = set(comp.FEATURE_NAMES)
    for g in groups:
        for name in V3_GROUPS.get(g, []):
            if name not in seen and name in v3.FEATURE_NAMES:
                extra.append(name)
                seen.add(name)
    return list(comp.FEATURE_NAMES) + extra


# Module-level default used by miner unless artifact overrides names.
FEATURE_NAMES: list[str] = feature_names_for_groups()


def extract_chunk_features(
    hands: list[dict] | None,
    *,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    names = feature_names or FEATURE_NAMES
    feat = dict(comp.extract_chunk_features(hands))
    v3_feat = v3.extract_chunk_features(hands)
    feat.update(v3_feat)
    return {name: float(feat.get(name, 0.0)) for name in names}


def features_to_vector(
    feat: dict[str, float],
    *,
    feature_names: list[str] | None = None,
) -> list[float]:
    names = feature_names or FEATURE_NAMES
    return [float(feat.get(name, 0.0)) for name in names]
