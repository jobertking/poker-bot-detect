"""Deterministic chunk-group features for Poker44 bot detection (v3).

Adds sequence bigrams, button-relative seat behavior, sizing regularity,
and cross-hand consistency. Miner-visible fields only.
No hand_id / chunkId / dates / hashes / player_uid features.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean, median, pstdev
from typing import Any, Iterable

ACTION_TYPES = ("fold", "call", "check", "bet", "raise")
STREETS = ("preflop", "flop", "turn", "river")
BIGRAMS = (
    "fold_fold",
    "raise_fold",
    "bet_fold",
    "raise_call",
    "bet_call",
    "check_bet",
    "check_raise",
    "call_raise",
    "raise_raise",
    "bet_raise",
    "check_check",
    "call_call",
)


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return float(mean(values)) if values else default


def _safe_median(values: list[float], default: float = 0.0) -> float:
    return float(median(values)) if values else default


def _safe_std(values: list[float], default: float = 0.0) -> float:
    if len(values) < 2:
        return default
    return float(pstdev(values))


def _safe_pct(values: list[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, int(round(q * (len(arr) - 1)))))
    return float(arr[idx])


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _entropy(counts: dict[str, int] | Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _safe_mean(values)
    if abs(m) < 1e-9:
        return 0.0
    return _safe_std(values) / abs(m)


def _seat_rel_to_button(seat: Any, button: Any, max_seats: int) -> int | None:
    try:
        s = int(seat)
        b = int(button)
        n = max(int(max_seats), 2)
        return (s - b) % n
    except (TypeError, ValueError):
        return None


def _action_counts(actions: Iterable[dict]) -> dict[str, int]:
    counts = {kind: 0 for kind in ACTION_TYPES}
    for action in actions:
        kind = action.get("action_type")
        if kind in counts:
            counts[kind] += 1
    return counts


def _hand_features(hand: dict) -> dict[str, float]:
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]
    players = [p for p in (hand.get("players") or []) if isinstance(p, dict)]
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}
    metadata = hand.get("metadata") or {}

    counts = _action_counts(actions)
    meaningful = max(1, sum(counts.values()))
    button = metadata.get("button_seat")
    max_seats = metadata.get("max_seats") or 6

    street_action_counts = {s: 0 for s in STREETS}
    street_agg = {s: 0 for s in STREETS}
    street_amounts: dict[str, list[float]] = {s: [] for s in STREETS}
    amounts_bb: list[float] = []
    pots_before: list[float] = []
    pots_after: list[float] = []
    aggression_actions = 0
    passive_actions = 0
    raise_sizes: list[float] = []
    zero_size_actions = 0
    sized_actions = 0
    round_size_hits = 0
    fold_vs_agg = 0
    agg_opportunities = 0
    same_action_streak = 0
    prev_kind = None
    aggression_streaks = 0
    kinds_seq: list[str] = []
    ip_actions = 0
    oop_actions = 0
    button_actions = 0
    seat_action_counts: Counter[int] = Counter()

    for action in actions:
        street = action.get("street")
        kind = str(action.get("action_type") or "")
        if kind in ACTION_TYPES:
            kinds_seq.append(kind)
        if street in street_action_counts:
            street_action_counts[street] += 1
            if kind in ("bet", "raise"):
                street_agg[street] += 1

        amt = action.get("normalized_amount_bb")
        if amt is not None:
            amt_f = _as_float(amt)
            amounts_bb.append(amt_f)
            sized_actions += 1
            if amt_f <= 1e-9:
                zero_size_actions += 1
            # Bots often use round BB sizes (2, 2.5, 3, 5, ...).
            if abs(amt_f - round(amt_f * 2) / 2) < 1e-6 and amt_f > 0:
                round_size_hits += 1
            if street in street_amounts:
                street_amounts[street].append(amt_f)
            if kind in ("bet", "raise"):
                raise_sizes.append(amt_f)

        pb = action.get("pot_before")
        if pb is not None:
            pots_before.append(_as_float(pb))
        pa = action.get("pot_after")
        if pa is not None:
            pots_after.append(_as_float(pa))

        if kind in ("bet", "raise"):
            aggression_actions += 1
            if prev_kind in ("bet", "raise"):
                aggression_streaks += 1
        elif kind in ("call", "check"):
            passive_actions += 1

        if prev_kind in ("bet", "raise"):
            agg_opportunities += 1
            if kind == "fold":
                fold_vs_agg += 1
        if prev_kind is not None and kind == prev_kind:
            same_action_streak += 1

        rel = _seat_rel_to_button(action.get("actor_seat"), button, max_seats)
        if rel is not None:
            seat_action_counts[rel] += 1
            if rel == 0:
                button_actions += 1
            # crude IP/OOP: seats closer after button tend toward IP on many streets
            if rel in (0, 1, 2):
                ip_actions += 1
            else:
                oop_actions += 1

        prev_kind = kind

    # Bigrams
    bigram_counts = {b: 0 for b in BIGRAMS}
    for a, b in zip(kinds_seq, kinds_seq[1:]):
        key = f"{a}_{b}"
        if key in bigram_counts:
            bigram_counts[key] += 1
    n_bigrams = max(1, len(kinds_seq) - 1)

    starting_stacks = [
        _as_float(p.get("starting_stack"))
        for p in players
        if p.get("starting_stack") is not None
    ]
    showed = sum(1 for p in players if p.get("showed_hand"))

    ended = str(metadata.get("hand_ended_on_street") or "").lower()
    ended_depth = {
        "preflop": 0.0,
        "flop": 1.0 / 3.0,
        "turn": 2.0 / 3.0,
        "river": 1.0,
    }.get(ended, _ratio(len(streets), 3.0))

    total_street_actions = max(1, sum(street_action_counts.values()))
    pot_growth = 0.0
    if pots_before and pots_after:
        pot_growth = max(0.0, pots_after[-1] - pots_before[0])
    pot_growth_rate = _ratio(pot_growth, max(_safe_mean(pots_before), 1e-6))

    mean_stack = _safe_mean(starting_stacks)
    mean_pot = _safe_mean(pots_before)
    unique_sizes = len({round(a, 2) for a in amounts_bb if a > 0})
    seat_entropy = _entropy(seat_action_counts)

    feats = {
        "n_actions": float(len(actions)),
        "n_players": float(len(players)),
        "n_streets": float(len(streets)),
        "street_depth": _ratio(len(streets), 3.0),
        "ended_depth": ended_depth,
        "ended_preflop": 1.0 if ended == "preflop" else 0.0,
        "ended_river": 1.0 if ended == "river" else 0.0,
        "showdown": 1.0 if outcome.get("showdown") else 0.0,
        "showed_hand_rate": _ratio(showed, max(1, len(players))),
        "fold_ratio": _ratio(counts["fold"], meaningful),
        "call_ratio": _ratio(counts["call"], meaningful),
        "check_ratio": _ratio(counts["check"], meaningful),
        "bet_ratio": _ratio(counts["bet"], meaningful),
        "raise_ratio": _ratio(counts["raise"], meaningful),
        "aggression_factor": _ratio(aggression_actions, max(1, passive_actions)),
        "aggression_share": _ratio(aggression_actions, meaningful),
        "action_entropy": _entropy(counts),
        "aggression_streak_rate": _ratio(aggression_streaks, max(1, len(actions) - 1)),
        "same_action_streak_rate": _ratio(same_action_streak, max(1, len(actions) - 1)),
        "fold_vs_aggression_rate": _ratio(fold_vs_agg, max(1, agg_opportunities)),
        "zero_size_rate": _ratio(zero_size_actions, max(1, sized_actions)),
        "round_size_rate": _ratio(round_size_hits, max(1, sized_actions)),
        "unique_size_rate": _ratio(unique_sizes, max(1, sized_actions)),
        "amount_cv": _cv(amounts_bb),
        "preflop_action_share": _ratio(street_action_counts["preflop"], total_street_actions),
        "flop_action_share": _ratio(street_action_counts["flop"], total_street_actions),
        "turn_action_share": _ratio(street_action_counts["turn"], total_street_actions),
        "river_action_share": _ratio(street_action_counts["river"], total_street_actions),
        "preflop_agg_share": _ratio(street_agg["preflop"], max(1, street_action_counts["preflop"])),
        "flop_agg_share": _ratio(street_agg["flop"], max(1, street_action_counts["flop"])),
        "turn_agg_share": _ratio(street_agg["turn"], max(1, street_action_counts["turn"])),
        "river_agg_share": _ratio(street_agg["river"], max(1, street_action_counts["river"])),
        "mean_normalized_amount_bb": _safe_mean(amounts_bb),
        "median_normalized_amount_bb": _safe_median(amounts_bb),
        "std_normalized_amount_bb": _safe_std(amounts_bb),
        "p10_normalized_amount_bb": _safe_pct(amounts_bb, 0.1),
        "p90_normalized_amount_bb": _safe_pct(amounts_bb, 0.9),
        "max_normalized_amount_bb": max(amounts_bb) if amounts_bb else 0.0,
        "mean_raise_size_bb": _safe_mean(raise_sizes),
        "std_raise_size_bb": _safe_std(raise_sizes),
        "raise_size_cv": _cv(raise_sizes),
        "mean_flop_amount_bb": _safe_mean(street_amounts["flop"]),
        "mean_turn_amount_bb": _safe_mean(street_amounts["turn"]),
        "mean_river_amount_bb": _safe_mean(street_amounts["river"]),
        "mean_pot_before": mean_pot,
        "std_pot_before": _safe_std(pots_before),
        "mean_pot_after": _safe_mean(pots_after),
        "pot_growth": pot_growth,
        "pot_growth_rate": pot_growth_rate,
        "mean_starting_stack": mean_stack,
        "std_starting_stack": _safe_std(starting_stacks),
        "stack_cv": _cv(starting_stacks),
        "stack_to_pot": _ratio(mean_stack, max(mean_pot, 1e-6)),
        "total_pot": _as_float((outcome or {}).get("total_pot")),
        "button_action_share": _ratio(button_actions, max(1, len(actions))),
        "ip_action_share": _ratio(ip_actions, max(1, ip_actions + oop_actions)),
        "seat_action_entropy": seat_entropy,
        **{f"bg_{k}": _ratio(v, n_bigrams) for k, v in bigram_counts.items()},
    }
    return feats


HAND_FEATURE_NAMES: list[str] = [
    "n_actions",
    "n_players",
    "n_streets",
    "street_depth",
    "ended_depth",
    "ended_preflop",
    "ended_river",
    "showdown",
    "showed_hand_rate",
    "fold_ratio",
    "call_ratio",
    "check_ratio",
    "bet_ratio",
    "raise_ratio",
    "aggression_factor",
    "aggression_share",
    "action_entropy",
    "aggression_streak_rate",
    "same_action_streak_rate",
    "fold_vs_aggression_rate",
    "zero_size_rate",
    "round_size_rate",
    "unique_size_rate",
    "amount_cv",
    "preflop_action_share",
    "flop_action_share",
    "turn_action_share",
    "river_action_share",
    "preflop_agg_share",
    "flop_agg_share",
    "turn_agg_share",
    "river_agg_share",
    "mean_normalized_amount_bb",
    "median_normalized_amount_bb",
    "std_normalized_amount_bb",
    "p10_normalized_amount_bb",
    "p90_normalized_amount_bb",
    "max_normalized_amount_bb",
    "mean_raise_size_bb",
    "std_raise_size_bb",
    "raise_size_cv",
    "mean_flop_amount_bb",
    "mean_turn_amount_bb",
    "mean_river_amount_bb",
    "mean_pot_before",
    "std_pot_before",
    "mean_pot_after",
    "pot_growth",
    "pot_growth_rate",
    "mean_starting_stack",
    "std_starting_stack",
    "stack_cv",
    "stack_to_pot",
    "total_pot",
    "button_action_share",
    "ip_action_share",
    "seat_action_entropy",
    *[f"bg_{k}" for k in BIGRAMS],
]


def extract_chunk_features(hands: list[dict] | None) -> dict[str, float]:
    hands = [h for h in (hands or []) if isinstance(h, dict)]
    base = {f"mean_{name}": 0.0 for name in HAND_FEATURE_NAMES}
    base.update({f"std_{name}": 0.0 for name in HAND_FEATURE_NAMES})
    base["hand_count"] = float(len(hands))
    base["empty_chunk"] = 1.0 if not hands else 0.0

    if not hands:
        # still expose chunk extras with zeros
        for name in _CHUNK_EXTRA_NAMES:
            base[name] = 0.0
        return base

    per_hand = [_hand_features(hand) for hand in hands]
    for name in HAND_FEATURE_NAMES:
        values = [hf[name] for hf in per_hand]
        base[f"mean_{name}"] = _safe_mean(values)
        base[f"std_{name}"] = _safe_std(values)

    all_actions: list[dict] = []
    all_kinds: list[str] = []
    for hand in hands:
        for a in hand.get("actions") or []:
            if not isinstance(a, dict):
                continue
            all_actions.append(a)
            kind = a.get("action_type")
            if kind in ACTION_TYPES:
                all_kinds.append(str(kind))

    amounts = [
        _as_float(a.get("normalized_amount_bb"))
        for a in all_actions
        if a.get("normalized_amount_bb") is not None
    ]
    street_counts = {s: 0 for s in STREETS}
    type_counts: Counter[str] = Counter()
    for action in all_actions:
        street = action.get("street")
        if street in street_counts:
            street_counts[street] += 1
        kind = action.get("action_type")
        if kind in ACTION_TYPES:
            type_counts[kind] += 1
    total_actions = max(1, sum(street_counts.values()))

    showdowns = sum(1 for h in hands if (h.get("outcome") or {}).get("showdown"))
    ended_preflop = sum(
        1
        for h in hands
        if str((h.get("metadata") or {}).get("hand_ended_on_street") or "").lower()
        == "preflop"
    )
    ended_river = sum(
        1
        for h in hands
        if str((h.get("metadata") or {}).get("hand_ended_on_street") or "").lower()
        == "river"
    )

    fold_ratios = [hf["fold_ratio"] for hf in per_hand]
    agg_factors = [hf["aggression_factor"] for hf in per_hand]
    amount_means = [hf["mean_normalized_amount_bb"] for hf in per_hand]
    entropies = [hf["action_entropy"] for hf in per_hand]
    round_rates = [hf["round_size_rate"] for hf in per_hand]
    n_actions_list = [hf["n_actions"] for hf in per_hand]
    street_depths = [hf["street_depth"] for hf in per_hand]

    bigram_counts = {b: 0 for b in BIGRAMS}
    for a, b in zip(all_kinds, all_kinds[1:]):
        key = f"{a}_{b}"
        if key in bigram_counts:
            bigram_counts[key] += 1
    n_bigrams = max(1, len(all_kinds) - 1)

    # Fingerprint diversity across hands: how often hand signatures repeat
    signatures = []
    for hf in per_hand:
        signatures.append(
            (
                round(hf["fold_ratio"], 2),
                round(hf["aggression_share"], 2),
                round(hf["street_depth"], 2),
                round(hf["mean_normalized_amount_bb"], 1),
            )
        )
    sig_counts = Counter(signatures)
    max_sig_share = max(sig_counts.values()) / max(1, len(signatures))

    base["chunk_showdown_rate"] = _ratio(showdowns, len(hands))
    base["chunk_ended_preflop_rate"] = _ratio(ended_preflop, len(hands))
    base["chunk_ended_river_rate"] = _ratio(ended_river, len(hands))
    base["chunk_mean_normalized_amount_bb"] = _safe_mean(amounts)
    base["chunk_std_normalized_amount_bb"] = _safe_std(amounts)
    base["chunk_amount_cv"] = _cv(amounts)
    base["chunk_p10_normalized_amount_bb"] = _safe_pct(amounts, 0.1)
    base["chunk_p90_normalized_amount_bb"] = _safe_pct(amounts, 0.9)
    base["chunk_unique_size_rate"] = _ratio(len({round(a, 2) for a in amounts if a > 0}), max(1, len(amounts)))
    base["chunk_flop_action_share"] = _ratio(street_counts["flop"], total_actions)
    base["chunk_turn_action_share"] = _ratio(street_counts["turn"], total_actions)
    base["chunk_river_action_share"] = _ratio(street_counts["river"], total_actions)
    base["chunk_preflop_action_share"] = _ratio(street_counts["preflop"], total_actions)
    base["chunk_n_actions"] = float(len(all_actions))
    base["chunk_action_entropy"] = _entropy(dict(type_counts))
    base["chunk_fold_ratio_std"] = _safe_std(fold_ratios)
    base["chunk_aggression_std"] = _safe_std(agg_factors)
    base["chunk_amount_mean_std"] = _safe_std(amount_means)
    base["chunk_entropy_std"] = _safe_std(entropies)
    base["chunk_n_actions_std"] = _safe_std(n_actions_list)
    base["chunk_street_depth_std"] = _safe_std(street_depths)
    base["chunk_fold_ratio_mean"] = _safe_mean(fold_ratios)
    base["chunk_aggression_mean"] = _safe_mean(agg_factors)
    base["chunk_round_size_mean"] = _safe_mean(round_rates)
    base["chunk_hand_signature_max_share"] = float(max_sig_share)
    base["chunk_hand_signature_entropy"] = _entropy(sig_counts)
    for k, v in bigram_counts.items():
        base[f"chunk_bg_{k}"] = _ratio(v, n_bigrams)
    return base


_CHUNK_EXTRA_NAMES = [
    "chunk_showdown_rate",
    "chunk_ended_preflop_rate",
    "chunk_ended_river_rate",
    "chunk_mean_normalized_amount_bb",
    "chunk_std_normalized_amount_bb",
    "chunk_amount_cv",
    "chunk_p10_normalized_amount_bb",
    "chunk_p90_normalized_amount_bb",
    "chunk_unique_size_rate",
    "chunk_flop_action_share",
    "chunk_turn_action_share",
    "chunk_river_action_share",
    "chunk_preflop_action_share",
    "chunk_n_actions",
    "chunk_action_entropy",
    "chunk_fold_ratio_std",
    "chunk_aggression_std",
    "chunk_amount_mean_std",
    "chunk_entropy_std",
    "chunk_n_actions_std",
    "chunk_street_depth_std",
    "chunk_fold_ratio_mean",
    "chunk_aggression_mean",
    "chunk_round_size_mean",
    "chunk_hand_signature_max_share",
    "chunk_hand_signature_entropy",
    *[f"chunk_bg_{k}" for k in BIGRAMS],
]


FEATURE_NAMES: list[str] = (
    ["hand_count", "empty_chunk"]
    + [f"mean_{name}" for name in HAND_FEATURE_NAMES]
    + [f"std_{name}" for name in HAND_FEATURE_NAMES]
    + _CHUNK_EXTRA_NAMES
)


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]
