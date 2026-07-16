"""Synthesize live-sized (80–100 hand) training chunks from public benchmark.

Live validators typically send ~80–100 hands per chunk while the public
benchmark averages ~30–40. Same-label merges close that train/serve gap.

Inspired by open techniques in peer MIT miners; reimplemented for this repo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LargeAugmentationConfig:
    """Ratios are relative to the number of original training chunks."""

    large_ratio: float = 1.25
    medium_ratio: float = 0.40
    sources_per_merge: int = 3
    large_min_hands: int = 80
    large_max_hands: int = 100
    medium_min_hands: int = 50
    medium_max_hands: int = 75

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _balanced_labels(count: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(([0, 1] * ((count + 1) // 2))[:count], dtype=int)
    rng.shuffle(labels)
    return labels


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

    by_label = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    for label, indices in by_label.items():
        if len(indices) < sources_per_merge:
            raise ValueError(
                f"need at least {sources_per_merge} label={label} source chunks, got {len(indices)}"
            )

    merged: List[List[Dict[str, Any]]] = []
    merged_labels: List[int] = []
    merged_dates: List[str] = []
    for label in _balanced_labels(count, rng):
        candidates = by_label[int(label)]
        pool: List[Dict[str, Any]] = []
        selected = np.asarray([], dtype=int)
        for _ in range(64):
            selected = rng.choice(candidates, size=sources_per_merge, replace=False)
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
        target = min(len(pool), int(rng.integers(min_hands, max_hands + 1)))
        chosen = rng.choice(len(pool), size=target, replace=False)
        merged.append([pool[int(i)] for i in chosen])
        merged_labels.append(int(label))
        parent_dates = [date_list[int(i)] for i in selected if date_list[int(i)]]
        merged_dates.append(max(parent_dates) if parent_dates else "")
    return merged, np.asarray(merged_labels, dtype=int), merged_dates


def build_training_views(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    labels: np.ndarray,
    dates: Sequence[str],
    config: LargeAugmentationConfig,
    seed: int,
) -> Tuple[List[List[Dict[str, Any]]], np.ndarray, List[str], Dict[str, int]]:
    """Original + large (80–100) + medium (50–75) views for one fold."""

    original = [list(chunk) for chunk in chunks]
    labels = np.asarray(labels, dtype=int)
    date_list = [str(d) for d in dates]
    rng = np.random.default_rng(seed)
    large, y_large, d_large = generate_merged_chunks(
        original,
        labels,
        date_list,
        count=int(round(len(original) * max(0.0, config.large_ratio))),
        min_hands=config.large_min_hands,
        max_hands=config.large_max_hands,
        sources_per_merge=config.sources_per_merge,
        rng=rng,
    )
    medium, y_medium, d_medium = generate_merged_chunks(
        original,
        labels,
        date_list,
        count=int(round(len(original) * max(0.0, config.medium_ratio))),
        min_hands=config.medium_min_hands,
        max_hands=config.medium_max_hands,
        sources_per_merge=config.sources_per_merge,
        rng=rng,
    )
    full_chunks = original + large + medium
    full_labels = np.concatenate([labels, y_large, y_medium])
    full_dates = date_list + d_large + d_medium
    stats = {
        "original": len(original),
        "large": len(large),
        "medium": len(medium),
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
