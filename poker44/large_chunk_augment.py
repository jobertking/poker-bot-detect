"""Synthesize live-sized training chunks from public benchmark.

Live validators send variable 70–160 hand chunks (often 80–100, but Round 5
used many 110–160). Same-label merges close that train/serve gap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import math
import numpy as np

# Observed live distribution from logs/requests (Jul 2026 scan):
# ~80% of chunks are 80–100 hands; 101–160 is a minority (~12–20%).
LIVE_HAND_BUCKETS: tuple[tuple[int, int, float], ...] = (
    (70, 79, 0.03),
    (80, 100, 0.80),
    (101, 120, 0.07),
    (121, 140, 0.05),
    (141, 160, 0.05),
)


@dataclass(frozen=True)
class LargeAugmentationConfig:
    """Ratios are relative to the number of original training chunks."""

    large_ratio: float = 1.00
    medium_ratio: float = 0.30
    small_live_ratio: float = 0.25
    xlarge_ratio: float = 0.20
    sources_per_merge: int = 3
    large_min_hands: int = 80
    large_max_hands: int = 100
    medium_min_hands: int = 50
    medium_max_hands: int = 75
    small_live_min_hands: int = 70
    small_live_max_hands: int = 79
    xlarge_min_hands: int = 101
    xlarge_max_hands: int = 160

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _balanced_labels(count: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(([0, 1] * ((count + 1) // 2))[:count], dtype=int)
    rng.shuffle(labels)
    return labels


def _pick_bucket(rng: np.random.Generator) -> tuple[int, int]:
    weights = np.asarray([b[2] for b in LIVE_HAND_BUCKETS], dtype=np.float64)
    weights /= weights.sum()
    idx = int(rng.choice(len(LIVE_HAND_BUCKETS), p=weights))
    lo, hi, _ = LIVE_HAND_BUCKETS[idx]
    return lo, hi


def generate_merged_chunks(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    labels: np.ndarray,
    dates: Sequence[str] | None = None,
    *,
    count: int,
    min_hands: int,
    max_hands: int,
    sources_per_merge: int,
    rng: np.random.Generator,
    target_hands: Sequence[int] | None = None,
) -> Tuple[List[List[Dict[str, Any]]], np.ndarray, List[str]]:
    """Same-label mixtures. Dates become max(parent dates) when provided."""

    labels = np.asarray(labels, dtype=int)
    date_list = [str(d) for d in dates] if dates is not None else [""] * len(labels)
    if len(date_list) != len(labels):
        raise ValueError("dates length must match labels")
    if count <= 0:
        return [], np.zeros(0, dtype=int), []
    if min_hands <= 0 or min_hands > max_hands:
        raise ValueError("invalid merged hand range")
    if sources_per_merge < 2:
        raise ValueError("sources_per_merge must be at least 2")

    # Benchmark chunks are ~30–40 hands; scale merges for 120–160 hand targets.
    merge_sources = max(sources_per_merge, int(math.ceil(min_hands / 35)) + 1)

    by_label = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    for label, indices in by_label.items():
        if len(indices) < merge_sources:
            raise ValueError(
                f"need at least {merge_sources} label={label} source chunks, got {len(indices)}"
            )

    merged: List[List[Dict[str, Any]]] = []
    merged_labels: List[int] = []
    merged_dates: List[str] = []
    for i, label in enumerate(_balanced_labels(count, rng)):
        candidates = by_label[int(label)]
        pool: List[Dict[str, Any]] = []
        selected = np.asarray([], dtype=int)
        for _ in range(64):
            selected = rng.choice(candidates, size=merge_sources, replace=False)
            pool = [
                hand
                for idx in selected
                for hand in chunks[int(idx)]
                if isinstance(hand, dict)
            ]
            if len(pool) >= min_hands:
                break
        if len(pool) < min_hands:
            raise ValueError(
                f"could not build label={int(label)} merged chunk with at least {min_hands} hands"
            )
        if target_hands is not None and i < len(target_hands):
            target = int(target_hands[i])
            target = max(min_hands, min(max_hands, target, len(pool)))
        else:
            target = min(len(pool), int(rng.integers(min_hands, max_hands + 1)))
        chosen = rng.choice(len(pool), size=target, replace=False)
        merged.append([pool[int(j)] for j in chosen])
        merged_labels.append(int(label))
        parent_dates = [date_list[int(idx)] for idx in selected if date_list[int(idx)]]
        merged_dates.append(max(parent_dates) if parent_dates else "")
    return merged, np.asarray(merged_labels, dtype=int), merged_dates


def build_training_views(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    labels: np.ndarray,
    dates: Sequence[str],
    config: LargeAugmentationConfig,
    seed: int,
) -> Tuple[List[List[Dict[str, Any]]], np.ndarray, List[str], Dict[str, int]]:
    """Original + live-shaped views (70–79, 80–100, 101–160) + medium (50–75)."""

    original = [list(chunk) for chunk in chunks]
    labels = np.asarray(labels, dtype=int)
    date_list = [str(d) for d in dates]
    rng = np.random.default_rng(seed)

    def _gen(count: int, lo: int, hi: int) -> tuple[list, np.ndarray, list]:
        if count <= 0:
            return [], np.zeros(0, dtype=int), []
        return generate_merged_chunks(
            original,
            labels,
            date_list,
            count=count,
            min_hands=lo,
            max_hands=hi,
            sources_per_merge=config.sources_per_merge,
            rng=rng,
        )

    n = len(original)
    large, y_large, d_large = _gen(
        int(round(n * max(0.0, config.large_ratio))),
        config.large_min_hands,
        config.large_max_hands,
    )
    medium, y_medium, d_medium = _gen(
        int(round(n * max(0.0, config.medium_ratio))),
        config.medium_min_hands,
        config.medium_max_hands,
    )
    small, y_small, d_small = _gen(
        int(round(n * max(0.0, config.small_live_ratio))),
        config.small_live_min_hands,
        config.small_live_max_hands,
    )
    xlarge, y_xlarge, d_xlarge = _gen(
        int(round(n * max(0.0, config.xlarge_ratio))),
        config.xlarge_min_hands,
        config.xlarge_max_hands,
    )

    full_chunks = original + large + medium + small + xlarge
    full_labels = np.concatenate([labels, y_large, y_medium, y_small, y_xlarge])
    full_dates = date_list + d_large + d_medium + d_small + d_xlarge
    stats = {
        "original": len(original),
        "large": len(large),
        "medium": len(medium),
        "small_live": len(small),
        "xlarge": len(xlarge),
        "full_total": len(full_chunks),
    }
    return full_chunks, full_labels, full_dates, stats


def build_live_shaped_holdout(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    labels: np.ndarray,
    dates: Sequence[str],
    *,
    seed: int,
    ratio: float = 1.0,
    min_hands: int = 80,
    max_hands: int = 100,
    sources_per_merge: int = 3,
) -> Tuple[List[List[Dict[str, Any]]], np.ndarray, List[str]]:
    """Build 80–100 hand holdout views for live-shaped offline scoring only."""

    count = int(round(len(chunks) * max(0.0, ratio)))
    if count <= 0:
        return [], np.zeros(0, dtype=int), []
    rng = np.random.default_rng(seed)
    return generate_merged_chunks(
        chunks,
        labels,
        dates,
        count=count,
        min_hands=min_hands,
        max_hands=max_hands,
        sources_per_merge=sources_per_merge,
        rng=rng,
    )


def build_live_distribution_holdout(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    labels: np.ndarray,
    dates: Sequence[str],
    *,
    seed: int,
    ratio: float = 1.0,
    sources_per_merge: int = 3,
) -> Tuple[List[List[Dict[str, Any]]], np.ndarray, List[str]]:
    """Holdout with hand counts sampled from observed live bucket distribution (70–160)."""

    count = int(round(len(chunks) * max(0.0, ratio)))
    if count <= 0:
        return [], np.zeros(0, dtype=int), []
    rng = np.random.default_rng(seed)
    targets = []
    bucket_ranges = []
    for _ in range(count):
        lo, hi = _pick_bucket(rng)
        bucket_ranges.append((lo, hi))
        targets.append(int(rng.integers(lo, hi + 1)))

    by_label = {label: np.flatnonzero(np.asarray(labels, dtype=int) == label) for label in (0, 1)}
    date_list = [str(d) for d in dates]
    merged: List[List[Dict[str, Any]]] = []
    merged_labels: List[int] = []
    merged_dates: List[str] = []

    for i, label in enumerate(_balanced_labels(count, rng)):
        lo, hi = bucket_ranges[i]
        target = targets[i]
        merge_sources = max(sources_per_merge, int(math.ceil(lo / 35)) + 1)
        candidates = by_label[int(label)]
        if len(candidates) < merge_sources:
            raise ValueError(
                f"need at least {merge_sources} label={int(label)} source chunks, got {len(candidates)}"
            )
        pool: List[Dict[str, Any]] = []
        selected = np.asarray([], dtype=int)
        for _ in range(64):
            selected = rng.choice(candidates, size=merge_sources, replace=False)
            pool = [
                hand
                for idx in selected
                for hand in chunks[int(idx)]
                if isinstance(hand, dict)
            ]
            if len(pool) >= lo:
                break
        if len(pool) < lo:
            raise ValueError(f"could not build holdout chunk with >= {lo} hands")
        take = max(lo, min(hi, target, len(pool)))
        chosen = rng.choice(len(pool), size=take, replace=False)
        merged.append([pool[int(j)] for j in chosen])
        merged_labels.append(int(label))
        parent_dates = [date_list[int(idx)] for idx in selected if date_list[int(idx)]]
        merged_dates.append(max(parent_dates) if parent_dates else "")

    return merged, np.asarray(merged_labels, dtype=int), merged_dates
