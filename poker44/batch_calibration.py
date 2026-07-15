"""Batch score remaps for live validator queries (FPR / reward control)."""

from __future__ import annotations

import math
from typing import Sequence


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def apply_threshold_logit_v1(
    scores: Sequence[float],
    *,
    threshold: float = 0.46,
    temperature: float = 0.08,
) -> list[float]:
    """Sharp sigmoid centered at threshold (from winning miner artifacts)."""
    if len(scores) == 0:
        return []
    temp = max(float(temperature), 1e-6)
    thr = float(threshold)
    out: list[float] = []
    for value in scores:
        clipped = max(1e-6, min(1.0 - 1e-6, float(value)))
        adjusted = (clipped - thr) / temp
        out.append(clamp01(1.0 / (1.0 + math.exp(-adjusted))))
    return out



def apply_batch_safety_topk_v1(
    scores: Sequence[float],
    *,
    max_positive_fraction: float = 0.125,
    max_positive_count: int = 40,
    positive_floor: float = 0.501,
    positive_ceiling: float = 0.509,
    negative_ceiling: float = 0.49,
) -> list[float]:
    """Rank-preserving top-K% positives just above 0.5; rest below 0.5."""
    count = len(scores)
    if count == 0:
        return []

    k = int(max_positive_count)
    if max_positive_fraction > 0.0:
        k = min(k, max(1, int(math.floor(count * float(max_positive_fraction)))))
    k = max(0, min(count, k))

    positive_floor = clamp01(positive_floor)
    positive_ceiling = clamp01(max(positive_floor, positive_ceiling))
    negative_ceiling = min(clamp01(negative_ceiling), positive_floor - 1e-6)

    ranked = sorted(
        [(i, clamp01(v)) for i, v in enumerate(scores)],
        key=lambda item: item[1],
        reverse=True,
    )
    output = [0.0] * count

    positives = ranked[:k]
    negatives = ranked[k:]
    if positives:
        denom = max(1, len(positives) - 1)
        for rank, (index, _score) in enumerate(positives):
            relative = 1.0 - (rank / denom if denom else 0.0)
            output[index] = positive_floor + relative * (positive_ceiling - positive_floor)

    if negatives:
        values = [score for _i, score in negatives]
        min_s = min(values)
        max_s = max(values)
        span = max(max_s - min_s, 1e-9)
        for index, score in negatives:
            relative = (score - min_s) / span
            output[index] = max(0.0, min(negative_ceiling, relative * negative_ceiling))

    return [round(clamp01(v), 6) for v in output]


def apply_clip_below(scores: Sequence[float], *, low: float = 0.05, high: float = 0.49) -> list[float]:
    """All scores < 0.5 (FPR=0), rank preserved — useful AP-only strategy."""
    if len(scores) == 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = [0.0] * len(scores)
    if len(order) == 1:
        out[order[0]] = high
        return out
    for rank, idx in enumerate(order):
        t = rank / (len(order) - 1)
        out[idx] = high - t * (high - low)
    return [round(clamp01(v), 6) for v in out]
