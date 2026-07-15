"""Competitive feature schema adapted from top SN126 miners (signatures + quantiles)."""

from __future__ import annotations

from features.competitive_features import chunk_features

# Deterministic feature order from a non-empty synthetic call.
def _build_feature_names() -> list[str]:
    probe = chunk_features(
        [
            {
                "metadata": {
                    "max_seats": 6,
                    "hero_seat": 1,
                    "button_seat": 1,
                    "hand_ended_on_street": "flop",
                },
                "players": [{"seat": 1, "starting_stack": 2.0}],
                "streets": [{"street": "flop"}],
                "actions": [
                    {
                        "action_type": "raise",
                        "actor_seat": 1,
                        "street": "preflop",
                        "normalized_amount_bb": 2.5,
                        "pot_before": 0.03,
                        "pot_after": 0.08,
                    },
                    {
                        "action_type": "call",
                        "actor_seat": 2,
                        "street": "preflop",
                        "normalized_amount_bb": 2.5,
                        "pot_before": 0.08,
                        "pot_after": 0.13,
                    },
                ],
                "outcome": {"showdown": False, "total_pot": 0.13},
            }
        ]
    )
    return sorted(probe.keys())


FEATURE_NAMES: list[str] = _build_feature_names()


def extract_chunk_features(hands: list[dict] | None) -> dict[str, float]:
    feat = chunk_features(list(hands or []))
    # Ensure full width even for empty chunks.
    return {name: float(feat.get(name, 0.0)) for name in FEATURE_NAMES}


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]
