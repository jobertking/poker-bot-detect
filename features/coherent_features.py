"""Hand-count-invariant cross-hand coherence features for Poker44 chunks.

Motivation: live validators send ~80-100 hands/chunk while the public benchmark
is ~30-40. Aggregate/count features drift with chunk size. These features are
computed per complete hand, then summarized across hands with distribution
statistics and action-signature summaries, so they stay stable regardless of
how many hands a chunk contains.

Miner-visible fields only. Hand order is never used (rows are sorted before
reduction). Action order *within* a hand is retained because repeated
action/actor/street/amount sequences are behavioral bot signals.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_AMOUNT_BOUNDS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
_DIST_STATS = ("mean", "std", "mad", "q10", "q25", "q50", "q75", "q90")
_SIGNATURE_KINDS = ("action", "actor", "street", "amount", "street_action", "full")
_SIGNATURE_STATS = (
    "top1_share",
    "top2_share",
    "unique_rate",
    "singleton_share",
    "entropy",
    "repeat_pair_rate",
)

_PER_HAND_FEATURE_NAMES = (
    "pot_before_mean_bb",
    "pot_before_max_bb",
    "pot_after_mean_bb",
    "pot_after_max_bb",
    "pot_after_final_bb",
    "pot_change_abs_mean_bb",
    "pot_delta_positive_mean_bb",
    "pot_growth_bb",
    "pot_monotonic_rate",
    "stack_mean_bb",
    "stack_std_bb",
    "stack_range_bb",
    "hero_stack_bb",
    "hero_stack_to_mean",
    "action_count",
    "action_type_unique",
    "actor_unique",
    "street_unique",
    "actor_switch_rate",
    "action_run_max_share",
    "actor_run_max_share",
    "action_entropy",
    "actor_entropy",
    "street_entropy",
    "preflop_share",
    "postflop_share",
    "blind_share",
    "allin_share",
    "aggressive_share",
    "passive_share",
    "amount_mean_bb",
    "amount_std_bb",
    "amount_q90_bb",
    "amount_max_bb",
    "amount_nonzero_share",
    "player_count",
    "seat_utilization",
    "hero_seat_norm",
    "button_seat_norm",
    "hero_button_distance_norm",
    "hero_button_same",
    "button_action_share",
    "hero_action_count",
    "hero_action_share",
    "hero_aggressive_share",
    "hero_fold_share",
    "raise_to_count",
    "raise_to_share",
    "raise_to_mean_bb",
    "raise_to_max_bb",
    "call_to_count",
    "call_to_share",
    "call_to_mean_bb",
    "call_to_max_bb",
)


def _build_feature_names() -> list[str]:
    names = [
        f"coherent__dist__{name}__{stat}"
        for name in _PER_HAND_FEATURE_NAMES
        for stat in _DIST_STATS
    ]
    names.extend(
        f"coherent__signature__{kind}__{stat}"
        for kind in _SIGNATURE_KINDS
        for stat in _SIGNATURE_STATS
    )
    return sorted(names)


COHERENT_FEATURE_NAMES: list[str] = _build_feature_names()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(result):
        return default
    return float(min(1_000_000.0, max(-1_000_000.0, result)))


def _integer(value: Any, default: int = 0) -> int:
    result = _number(value, float(default))
    try:
        return int(result)
    except (TypeError, ValueError, OverflowError):
        return default


def _token(value: Any) -> str:
    if value is None:
        return "<missing>"
    result = str(value).strip().lower()
    return result[:48] if result else "<missing>"


def _positive_array(values: Sequence[Any]) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray([max(0.0, _number(v)) for v in values], dtype=np.float64)
    return np.clip(arr, 0.0, 1_000_000.0)


def _mean(arr: np.ndarray) -> float:
    return float(arr.mean()) if arr.size else 0.0


def _max(arr: np.ndarray) -> float:
    return float(arr.max()) if arr.size else 0.0


def _amount_bb(action: Mapping[str, Any], bb: float) -> float:
    normalized = action.get("normalized_amount_bb")
    if normalized is not None:
        return max(0.0, _number(normalized))
    return max(0.0, _number(action.get("amount"))) / bb


def _amount_bucket(amount_bb: float) -> int:
    return bisect_right(_AMOUNT_BOUNDS, max(0.0, amount_bb)) - 1


def _entropy_seq(values: Sequence[Hashable]) -> float:
    if len(values) <= 1:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=np.float64)
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-15)).sum() / math.log(counts.size))


def _max_run_share(values: Sequence[Hashable]) -> float:
    if not values:
        return 0.0
    longest = current = 1
    for prev, val in zip(values, values[1:]):
        if val == prev:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest / len(values)


def _hand_row_and_signatures(hand: Mapping[str, Any]):
    meta_obj = hand.get("metadata")
    metadata: Mapping[str, Any] = meta_obj if isinstance(meta_obj, Mapping) else {}
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, Mapping)]
    players = [p for p in (hand.get("players") or []) if isinstance(p, Mapping)]

    bb = max(1e-6, abs(_number(metadata.get("bb"), 0.02)))
    hero_seat = _integer(metadata.get("hero_seat"))
    button_seat = _integer(metadata.get("button_seat"))
    player_seats = [_integer(p.get("seat")) for p in players]
    valid_seats = [s for s in player_seats if s > 0]
    max_seats = max(
        1,
        _integer(metadata.get("max_seats"), 0),
        max(valid_seats, default=0),
        hero_seat,
        button_seat,
    )

    action_types = tuple(_token(a.get("action_type")) for a in actions)
    action_streets = tuple(_token(a.get("street")) for a in actions)
    actor_seats = tuple(_integer(a.get("actor_seat")) for a in actions)
    action_amounts = _positive_array([_amount_bb(a, bb) for a in actions])
    amount_buckets = tuple(_amount_bucket(v) for v in action_amounts)

    pot_before = _positive_array([a.get("pot_before") for a in actions]) / bb
    pot_after = _positive_array([a.get("pot_after") for a in actions]) / bb
    if pot_before.size and pot_after.size:
        pot_change_abs_mean = float(np.mean(np.abs(pot_after - pot_before)))
        pot_delta_positive_mean = float(np.mean(np.maximum(0.0, pot_after - pot_before)))
        pot_growth = float(max(0.0, pot_after.max() - pot_before.min()))
    else:
        pot_change_abs_mean = pot_delta_positive_mean = pot_growth = 0.0
    pot_monotonic_rate = (
        float(np.mean(np.diff(pot_after) >= -1e-9)) if pot_after.size > 1 else 0.0
    )

    stacks = _positive_array([p.get("starting_stack") for p in players]) / bb
    stack_mean = _mean(stacks)
    stack_std = float(stacks.std()) if stacks.size else 0.0
    stack_range = float(np.ptp(stacks)) if stacks.size else 0.0
    hero_stack = 0.0
    for player, seat in zip(players, player_seats):
        if hero_seat > 0 and seat == hero_seat:
            hero_stack = max(0.0, _number(player.get("starting_stack"))) / bb
            break

    n_actions = len(actions)
    hero_mask = [hero_seat > 0 and seat == hero_seat for seat in actor_seats]
    hero_types = [k for k, is_hero in zip(action_types, hero_mask) if is_hero]
    hero_action_count = len(hero_types)
    aggressive = {"bet", "raise"}
    passive = {"check", "call"}
    action_counts = Counter(action_types)
    preflop_count = sum(s == "preflop" for s in action_streets)
    postflop_count = sum(s not in {"<missing>", "preflop"} for s in action_streets)
    button_action_count = sum(button_seat > 0 and s == button_seat for s in actor_seats)

    raise_targets = _positive_array(
        [a.get("raise_to") for a in actions if a.get("raise_to") is not None]
    ) / bb
    call_targets = _positive_array(
        [a.get("call_to") for a in actions if a.get("call_to") is not None]
    ) / bb

    if hero_seat > 0 and button_seat > 0:
        clockwise = (hero_seat - button_seat) % max_seats
        counter = (button_seat - hero_seat) % max_seats
        hero_button_distance = min(clockwise, counter) / max_seats
    else:
        hero_button_distance = 0.0

    row = {
        "pot_before_mean_bb": _mean(pot_before),
        "pot_before_max_bb": _max(pot_before),
        "pot_after_mean_bb": _mean(pot_after),
        "pot_after_max_bb": _max(pot_after),
        "pot_after_final_bb": float(pot_after[-1]) if pot_after.size else 0.0,
        "pot_change_abs_mean_bb": pot_change_abs_mean,
        "pot_delta_positive_mean_bb": pot_delta_positive_mean,
        "pot_growth_bb": pot_growth,
        "pot_monotonic_rate": pot_monotonic_rate,
        "stack_mean_bb": stack_mean,
        "stack_std_bb": stack_std,
        "stack_range_bb": stack_range,
        "hero_stack_bb": hero_stack,
        "hero_stack_to_mean": hero_stack / max(stack_mean, 1e-6) if hero_stack > 0 else 0.0,
        "action_count": float(n_actions),
        "action_type_unique": float(len(set(action_types))),
        "actor_unique": float(len({s for s in actor_seats if s > 0})),
        "street_unique": float(len({s for s in action_streets if s != "<missing>"})),
        "actor_switch_rate": (
            float(np.mean(np.diff(np.asarray(actor_seats, dtype=np.int64)) != 0))
            if len(actor_seats) > 1
            else 0.0
        ),
        "action_run_max_share": _max_run_share(action_types),
        "actor_run_max_share": _max_run_share(actor_seats),
        "action_entropy": _entropy_seq(action_types),
        "actor_entropy": _entropy_seq(actor_seats),
        "street_entropy": _entropy_seq(action_streets),
        "preflop_share": preflop_count / max(1, n_actions),
        "postflop_share": postflop_count / max(1, n_actions),
        "blind_share": (
            action_counts["small_blind"] + action_counts["big_blind"] + action_counts["ante"]
        ) / max(1, n_actions),
        "allin_share": action_counts["all_in"] / max(1, n_actions),
        "aggressive_share": sum(k in aggressive for k in action_types) / max(1, n_actions),
        "passive_share": sum(k in passive for k in action_types) / max(1, n_actions),
        "amount_mean_bb": _mean(action_amounts),
        "amount_std_bb": float(action_amounts.std()) if action_amounts.size else 0.0,
        "amount_q90_bb": float(np.quantile(action_amounts, 0.90)) if action_amounts.size else 0.0,
        "amount_max_bb": _max(action_amounts),
        "amount_nonzero_share": float(np.mean(action_amounts > 0.0)) if action_amounts.size else 0.0,
        "player_count": float(len(players)),
        "seat_utilization": len(players) / max_seats,
        "hero_seat_norm": hero_seat / max_seats if hero_seat > 0 else 0.0,
        "button_seat_norm": button_seat / max_seats if button_seat > 0 else 0.0,
        "hero_button_distance_norm": hero_button_distance,
        "hero_button_same": float(hero_seat > 0 and hero_seat == button_seat),
        "button_action_share": button_action_count / max(1, n_actions),
        "hero_action_count": float(hero_action_count),
        "hero_action_share": hero_action_count / max(1, n_actions),
        "hero_aggressive_share": (
            sum(k in aggressive for k in hero_types) / max(1, hero_action_count)
        ),
        "hero_fold_share": hero_types.count("fold") / max(1, hero_action_count),
        "raise_to_count": float(raise_targets.size),
        "raise_to_share": float(raise_targets.size) / max(1, n_actions),
        "raise_to_mean_bb": _mean(raise_targets),
        "raise_to_max_bb": _max(raise_targets),
        "call_to_count": float(call_targets.size),
        "call_to_share": float(call_targets.size) / max(1, n_actions),
        "call_to_mean_bb": _mean(call_targets),
        "call_to_max_bb": _max(call_targets),
    }

    signatures: dict[str, Hashable] = {
        "action": action_types,
        "actor": actor_seats,
        "street": action_streets,
        "amount": amount_buckets,
        "street_action": tuple(zip(action_streets, action_types)),
        "full": tuple(zip(action_streets, actor_seats, action_types, amount_buckets)),
    }
    return row, signatures


def _add_distributions(rows: Sequence[Mapping[str, float]], output: dict[str, float]) -> None:
    matrix = np.asarray(
        [[_number(row[name]) for name in _PER_HAND_FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    if not matrix.size:
        matrix = np.zeros((1, len(_PER_HAND_FEATURE_NAMES)), dtype=np.float64)
    matrix.sort(axis=0)  # hand-order invariance
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    medians = np.median(matrix, axis=0)
    mads = np.median(np.abs(matrix - medians), axis=0)
    quantiles = np.quantile(matrix, _QUANTILES, axis=0)
    for col, name in enumerate(_PER_HAND_FEATURE_NAMES):
        prefix = f"coherent__dist__{name}__"
        output[prefix + "mean"] = float(means[col])
        output[prefix + "std"] = float(stds[col])
        output[prefix + "mad"] = float(mads[col])
        for q_index, quantile in enumerate(_QUANTILES):
            output[prefix + f"q{int(quantile * 100):02d}"] = float(quantiles[q_index, col])


def _add_signature_summary(kind: str, sigs: Sequence[Hashable], output: dict[str, float]) -> None:
    counts = sorted(Counter(sigs).values(), reverse=True)
    total = sum(counts)
    prefix = f"coherent__signature__{kind}__"
    if total <= 0:
        for stat in _SIGNATURE_STATS:
            output[prefix + stat] = 0.0
        return
    probabilities = np.asarray(counts, dtype=np.float64) / total
    if len(counts) <= 1:
        entropy = 0.0
    else:
        entropy = float(-(probabilities * np.log(probabilities + 1e-15)).sum() / math.log(len(counts)))
    repeat_denominator = total * (total - 1)
    repeat_pairs = sum(c * (c - 1) for c in counts)
    output[prefix + "top1_share"] = counts[0] / total
    output[prefix + "top2_share"] = sum(counts[:2]) / total
    output[prefix + "unique_rate"] = len(counts) / total
    output[prefix + "singleton_share"] = sum(c == 1 for c in counts) / total
    output[prefix + "entropy"] = entropy
    output[prefix + "repeat_pair_rate"] = (
        repeat_pairs / repeat_denominator if repeat_denominator > 0 else 0.0
    )


def coherent_feature_dict(hands: Sequence[Mapping[str, Any]] | None) -> dict[str, float]:
    """Hand-order-invariant cross-hand coherence features for one chunk."""
    rows: list[dict[str, float]] = []
    signatures: dict[str, list[Hashable]] = {kind: [] for kind in _SIGNATURE_KINDS}
    for raw in hands or []:
        hand = raw if isinstance(raw, Mapping) else {}
        row, hand_sigs = _hand_row_and_signatures(hand)
        rows.append(row)
        for kind in _SIGNATURE_KINDS:
            signatures[kind].append(hand_sigs[kind])
    output: dict[str, float] = {}
    _add_distributions(rows, output)
    for kind in _SIGNATURE_KINDS:
        _add_signature_summary(kind, signatures[kind], output)
    return {name: _number(output.get(name, 0.0)) for name in COHERENT_FEATURE_NAMES}
