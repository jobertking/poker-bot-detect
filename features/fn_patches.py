"""FN-targeted chunk cues: stealth bots look diverse like humans.

Focus on pairwise hand similarity, first/second-half drift, and
diversity-vs-concentration structure that separates stealth FNs weakly
from true positives but can help when combined with competitive schema.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _mean(xs: list[float]) -> float:
    return _safe_div(sum(xs), len(xs))


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(max(0.0, _mean([(x - m) * (x - m) for x in xs])))


def _entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent / math.log(len(counts))


def _action_sig(hand: dict) -> tuple:
    return tuple(
        (a.get("action_type"), a.get("street"))
        for a in (hand.get("actions") or [])
        if isinstance(a, dict)
    )


def _amt_sig(hand: dict) -> tuple:
    out = []
    for a in hand.get("actions") or []:
        if not isinstance(a, dict):
            continue
        v = _safe_float(a.get("normalized_amount_bb"), 0.0)
        out.append(round(v, 1))
    return tuple(out[:16])


def _amt_bucket(v: float) -> str:
    if v <= 0:
        return "z"
    if v <= 1.0:
        return "s"
    if v <= 2.5:
        return "m"
    return "l"


def _jaccard(a: tuple, b: tuple) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return _safe_div(len(sa & sb), len(sa | sb))


def _agg_share(hand: dict) -> float:
    acts = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]
    if not acts:
        return 0.0
    n = sum(1 for a in acts if a.get("action_type") in ("bet", "raise"))
    return n / len(acts)


def extract_fn_patch_features(hands: list[dict] | None) -> dict[str, float]:
    hands = [h for h in (hands or []) if isinstance(h, dict)]
    out = {
        "fn_pair_action_jaccard_mean": 0.0,
        "fn_pair_action_jaccard_std": 0.0,
        "fn_pair_amount_jaccard_mean": 0.0,
        "fn_pair_amount_jaccard_std": 0.0,
        "fn_unique_action_sig_share": 0.0,
        "fn_unique_amount_sig_share": 0.0,
        "fn_repeat_top2_action_share": 0.0,
        "fn_half_aggression_drift": 0.0,
        "fn_size_hist_entropy": 0.0,
        "fn_diversity_concentration_gap": 0.0,
        "fn_uniform_diversity": 0.0,
        "fn_quantized_despite_diverse": 0.0,
        "fn_hand_count": 0.0,
    }
    n = len(hands)
    out["fn_hand_count"] = float(n)
    if n == 0:
        return out

    asigs = [_action_sig(h) for h in hands]
    bsigs = [_amt_sig(h) for h in hands]
    lim = min(n, 24)
    aj, bj = [], []
    for i in range(lim):
        for j in range(i + 1, lim):
            aj.append(_jaccard(asigs[i], asigs[j]))
            bj.append(_jaccard(bsigs[i], bsigs[j]))
    out["fn_pair_action_jaccard_mean"] = _mean(aj)
    out["fn_pair_action_jaccard_std"] = _std(aj)
    out["fn_pair_amount_jaccard_mean"] = _mean(bj)
    out["fn_pair_amount_jaccard_std"] = _std(bj)

    unique_a = _safe_div(len(set(asigs)), n)
    unique_b = _safe_div(len(set(bsigs)), n)
    out["fn_unique_action_sig_share"] = unique_a
    out["fn_unique_amount_sig_share"] = unique_b
    top2 = sum(v for _, v in Counter(asigs).most_common(2))
    top2_share = _safe_div(top2, n)
    out["fn_repeat_top2_action_share"] = top2_share
    out["fn_diversity_concentration_gap"] = unique_a - top2_share
    # High unique with low top2 std -> "flat" diversity (more human/stealth)
    out["fn_uniform_diversity"] = unique_a * (1.0 - top2_share)

    mid = max(1, n // 2)
    a1 = _mean([_agg_share(h) for h in hands[:mid]])
    a2 = _mean([_agg_share(h) for h in hands[mid:]])
    out["fn_half_aggression_drift"] = abs(a1 - a2)

    buckets = []
    roundish = 0
    sized = 0
    for h in hands:
        for a in h.get("actions") or []:
            if not isinstance(a, dict):
                continue
            v = _safe_float(a.get("normalized_amount_bb"), 0.0)
            buckets.append(_amt_bucket(v))
            if v > 0:
                sized += 1
                # Near-integer or half-bb sizing
                if abs(v - round(v)) < 0.05 or abs(v * 2 - round(v * 2)) < 0.05:
                    roundish += 1
    out["fn_size_hist_entropy"] = _entropy(buckets)
    round_rate = _safe_div(roundish, max(1, sized))
    # Stealth bots: diverse action signatures but still quantized sizing.
    out["fn_quantized_despite_diverse"] = unique_a * round_rate
    return out


FEATURE_NAMES: list[str] = sorted(extract_fn_patch_features([]).keys())


def features_to_vector(feat: dict[str, float]) -> list[float]:
    return [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]
