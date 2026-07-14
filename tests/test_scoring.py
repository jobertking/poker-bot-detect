from __future__ import annotations

import numpy as np

from poker44.score.scoring import reward


def test_reward_rejects_all_negative_operating_point_on_mixed_window() -> None:
    labels = np.array([0, 0, 0, 1, 1], dtype=int)
    scores = np.array([0.04, 0.12, 0.20, 0.42, 0.49], dtype=float)

    value, metrics = reward(scores, labels)

    assert metrics["ap_score"] > 0
    assert metrics["positive_prediction_rate"] == 0
    assert metrics["hard_bot_recall"] == 0
    assert metrics["threshold_sanity_quality"] == 0
    assert metrics["human_safety_penalty"] == 0
    assert value == 0


def test_reward_preserves_ranked_score_with_valid_threshold_signal() -> None:
    labels = np.array([0, 0, 0, 1, 1], dtype=int)
    scores = np.array([0.04, 0.12, 0.20, 0.57, 0.82], dtype=float)

    value, metrics = reward(scores, labels)

    assert metrics["ap_score"] == 1
    assert metrics["hard_bot_recall"] == 1
    assert metrics["hard_fpr"] == 0
    assert metrics["threshold_sanity_quality"] == 1
    assert metrics["human_safety_penalty"] == 1
    assert metrics["calibration_quality"] == 1
    assert metrics["latency_quality"] == 1
    assert np.isclose(metrics["base_score"], 1)
    assert np.isclose(value, 1)
    assert value > 0


def test_reward_weights_recall_more_than_rank_only_signal() -> None:
    labels = np.array([0, 0, 0, 1, 1], dtype=int)
    scores = np.array([0.10, 0.20, 0.30, 0.40, 0.80], dtype=float)

    value, metrics = reward(scores, labels)

    expected = (
        0.35 * metrics["ap_score"]
        + 0.30 * metrics["bot_recall"]
        + 0.20 * metrics["human_safety_penalty"]
        + 0.10 * metrics["calibration_quality"]
        + 0.05 * metrics["latency_quality"]
    )
    assert metrics["hard_bot_recall"] == 0.5
    assert metrics["hard_fpr"] == 0
    assert value == expected


def test_reward_penalizes_high_threshold_false_positive_rate() -> None:
    labels = np.array([0, 0, 0, 0, 1], dtype=int)
    scores = np.array([0.61, 0.62, 0.63, 0.64, 0.9], dtype=float)

    value, metrics = reward(scores, labels)

    assert metrics["hard_bot_recall"] == 1
    assert metrics["hard_fpr"] == 1
    assert metrics["threshold_sanity_quality"] == 0
    assert metrics["human_safety_penalty"] == 0
    assert value == 0
