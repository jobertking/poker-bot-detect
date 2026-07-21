"""Hand-level action n-gram features for Poker44 bot detection.

Tokenize each hand into street+action+size-bucket tokens; count unigrams,
bigrams, trigrams, and position-action tokens. Counts are per-hand normalized
at chunk aggregation so features stay stable across 30-160 hand batches.

Adapted from Poker44-cold-poker1 / poker44-handngram-miner (MIT).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from features.competitive_features import _safe_div, _safe_float


NGRAM_ACTION_CODES = {
    "fold": "F",
    "call": "C",
    "raise": "R",
    "check": "K",
    "bet": "B",
    "action": "A",
    "all_in": "I",
}

NGRAM_VOCAB: tuple[str, ...] = (
    'fBm', 'fBm|fCm', 'fBm|fCm|tBp', 'fBm|fCm|tK0', 'fBm|fCs', 'fBm|fF0',
    'fBm|fF0|fF0', 'fBm|fRp', 'fBm|fRp|fF0', 'fBm|tBp', 'fBm|tK0', 'fBm|tK0|rK0',
    'fBo', 'fBo|fF0', 'fBp', 'fBp|fCm', 'fBp|fCm|tK0', 'fBp|fCp',
    'fBp|fF0', 'fBp|fF0|fF0', 'fBp|fRp', 'fBp|fRp|fF0', 'fBp|tBp', 'fBp|tK0',
    'fBs', 'fBs|fF0', 'fCm', 'fCm|tBm', 'fCm|tBp', 'fCm|tBp|tF0',
    'fCm|tK0', 'fCm|tK0|rBp', 'fCm|tK0|rK0', 'fCm|tK0|tBp', 'fCm|tK0|tK0', 'fCp',
    'fCp|tBp', 'fCp|tBp|tF0', 'fCp|tK0', 'fCs', 'fCs|tK0', 'fF0',
    'fF0|fF0', 'fF0|fF0|fF0', 'fF0|tBp', 'fF0|tK0', 'fK0', 'fK0|fBm',
    'fK0|fBm|fCm', 'fK0|fBm|fF0', 'fK0|fBm|fRp', 'fK0|fBp', 'fK0|fBp|fF0', 'fK0|fBs',
    'fK0|fBs|fF0', 'fK0|fCm', 'fK0|fCm|tK0', 'fK0|fF0', 'fK0|fF0|fF0', 'fK0|fK0',
    'fK0|fK0|tBm', 'fK0|fK0|tBp', 'fK0|fK0|tK0', 'fK0|fRp', 'fK0|rBp', 'fK0|rBp|rF0',
    'fK0|rK0', 'fK0|rK0|rK0', 'fK0|tBm', 'fK0|tBm|tF0', 'fK0|tBo', 'fK0|tBp',
    'fK0|tBp|rBp', 'fK0|tBp|rK0', 'fK0|tBp|tF0', 'fK0|tBs', 'fK0|tBs|tF0', 'fK0|tCm',
    'fK0|tF0', 'fK0|tF0|tF0', 'fK0|tK0', 'fK0|tK0|rBm', 'fK0|tK0|rBp', 'fK0|tK0|rK0',
    'fK0|tK0|tBm', 'fK0|tK0|tBp', 'fK0|tK0|tF0', 'fK0|tK0|tK0', 'fRp', 'fRp|fCp',
    'fRp|fF0', 'len', 'nseats', 'pCm', 'pCm|fBm', 'pCm|fBm|fCm',
    'pCm|fBm|fF0', 'pCm|fBp', 'pCm|fBp|fF0', 'pCm|fCm', 'pCm|fCm|tK0', 'pCm|fK0',
    'pCm|fK0|fBm', 'pCm|fK0|fBp', 'pCm|fK0|fCm', 'pCm|fK0|fF0', 'pCm|fK0|fK0', 'pCm|fK0|tBm',
    'pCm|fK0|tBp', 'pCm|fK0|tK0', 'pCm|pCm', 'pCm|pCs', 'pCm|pCs|fK0', 'pCm|pF0',
    'pCm|pF0|fBm', 'pCm|pF0|fBp', 'pCm|pF0|fK0', 'pCm|pF0|pCm', 'pCm|pF0|pCs', 'pCm|pF0|pF0',
    'pCm|pF0|pK0', 'pCm|pF0|pRo', 'pCm|pF0|pRp', 'pCm|pRo', 'pCm|pRo|pF0', 'pCm|pRp',
    'pCm|pRp|pF0', 'pCp', 'pCp|fBp', 'pCp|fK0', 'pCp|pF0', 'pCs',
    'pCs|fBm', 'pCs|fBm|fF0', 'pCs|fBp', 'pCs|fBp|fF0', 'pCs|fF0', 'pCs|fK0',
    'pCs|fK0|fBm', 'pCs|fK0|fBp', 'pCs|fK0|fF0', 'pCs|fK0|fK0', 'pCs|fK0|tK0', 'pCs|pCs',
    'pCs|pF0', 'pCs|pK0', 'pCs|pK0|fK0', 'pCs|pRo', 'pF0', 'pF0|fBm',
    'pF0|fBm|fCm', 'pF0|fBm|fF0', 'pF0|fBm|tK0', 'pF0|fBp', 'pF0|fBp|fF0', 'pF0|fCm',
    'pF0|fCm|tBp', 'pF0|fCm|tK0', 'pF0|fCp', 'pF0|fF0', 'pF0|fK0', 'pF0|fK0|fBm',
    'pF0|fK0|fBp', 'pF0|fK0|fF0', 'pF0|fK0|fK0', 'pF0|fK0|rK0', 'pF0|fK0|tBm', 'pF0|fK0|tBp',
    'pF0|fK0|tK0', 'pF0|fRp', 'pF0|pCm', 'pF0|pCm|fBm', 'pF0|pCm|fBp', 'pF0|pCm|fCm',
    'pF0|pCm|fK0', 'pF0|pCm|pCs', 'pF0|pCm|pF0', 'pF0|pCm|pRo', 'pF0|pCm|pRp', 'pF0|pCp',
    'pF0|pCs', 'pF0|pCs|fBm', 'pF0|pCs|fBp', 'pF0|pCs|fK0', 'pF0|pCs|pF0', 'pF0|pCs|pK0',
    'pF0|pCs|pRo', 'pF0|pF0', 'pF0|pF0|fBm', 'pF0|pF0|fBp', 'pF0|pF0|fCm', 'pF0|pF0|fK0',
    'pF0|pF0|pCm', 'pF0|pF0|pCs', 'pF0|pF0|pF0', 'pF0|pF0|pK0', 'pF0|pF0|pRo', 'pF0|pF0|pRp',
    'pF0|pK0', 'pF0|pK0|fK0', 'pF0|pRo', 'pF0|pRo|fBm', 'pF0|pRo|fK0', 'pF0|pRo|pCm',
    'pF0|pRo|pF0', 'pF0|pRo|pRo', 'pF0|pRp', 'pF0|pRp|fK0', 'pF0|pRp|pCm', 'pF0|pRp|pCp',
    'pF0|pRp|pF0', 'pF0|pRp|pRo', 'pK0', 'pK0|fBm', 'pK0|fBm|fF0', 'pK0|fBp',
    'pK0|fK0', 'pK0|fK0|fBp', 'pK0|fK0|fF0', 'pK0|fK0|fK0', 'pK0|fK0|tK0', 'pK0|pCs',
    'pK0|pCs|fK0', 'pK0|pRo', 'pK0|pRo|pCm', 'pK0|pRo|pF0', 'pRo', 'pRo|fBm',
    'pRo|fBp', 'pRo|fK0', 'pRo|pCm', 'pRo|pCm|fBm', 'pRo|pCm|fBp', 'pRo|pCm|fK0',
    'pRo|pCm|pCs', 'pRo|pCm|pF0', 'pRo|pCp', 'pRo|pCp|fK0', 'pRo|pCp|pF0', 'pRo|pCs',
    'pRo|pF0', 'pRo|pF0|fBm', 'pRo|pF0|fK0', 'pRo|pF0|pCm', 'pRo|pF0|pCs', 'pRo|pF0|pF0',
    'pRo|pF0|pRo', 'pRo|pF0|pRp', 'pRo|pRo', 'pRo|pRo|pF0', 'pRo|pRp', 'pRo|pRp|pF0',
    'pRp', 'pRp|fK0', 'pRp|pCm', 'pRp|pCm|fBm', 'pRp|pCm|fK0', 'pRp|pCm|pF0',
    'pRp|pCp', 'pRp|pCs', 'pRp|pF0', 'pRp|pF0|fK0', 'pRp|pF0|pCm', 'pRp|pF0|pCs',
    'pRp|pF0|pF0', 'pRp|pF0|pRo', 'pRp|pRo', 'pRp|pRo|pCm', 'pRp|pRo|pF0', 'pos0B',
    'pos0C', 'pos0F', 'pos0K', 'pos0R', 'pos1B', 'pos1C',
    'pos1F', 'pos1K', 'pos1R', 'pos2B', 'pos2C', 'pos2F',
    'pos2K', 'pos2R', 'pos3B', 'pos3C', 'pos3F', 'pos3K',
    'pos3R', 'pos4B', 'pos4C', 'pos4F', 'pos4K', 'pos4R',
    'pos5B', 'pos5C', 'pos5F', 'pos5K', 'pos5R', 'rBm',
    'rBm|rCm', 'rBm|rF0', 'rBp', 'rBp|rCp', 'rBp|rF0', 'rBs',
    'rBs|rF0', 'rCm', 'rCp', 'rCs', 'rF0', 'rK0',
    'rK0|rBp', 'rK0|rCp', 'rK0|rF0', 'rK0|rK0', 'rRp', 'rRp|rCp',
    'tBm', 'tBm|rBp', 'tBm|rK0', 'tBm|rK0|rK0', 'tBm|tCm', 'tBm|tF0',
    'tBo', 'tBp', 'tBp|rBp', 'tBp|rBp|rCp', 'tBp|rBp|rF0', 'tBp|rK0',
    'tBp|rK0|rK0', 'tBp|tCm', 'tBp|tCp', 'tBp|tCp|rBp', 'tBp|tF0', 'tBs',
    'tBs|tF0', 'tCm', 'tCm|rBp', 'tCm|rBp|rF0', 'tCm|rK0', 'tCm|rK0|rK0',
    'tCp', 'tCp|rBp', 'tCp|rBp|rF0', 'tCp|rK0', 'tCs', 'tCs|rK0',
    'tF0', 'tF0|tF0', 'tK0', 'tK0|rBm', 'tK0|rBm|rCm', 'tK0|rBm|rF0',
    'tK0|rBp', 'tK0|rBp|rCp', 'tK0|rBp|rF0', 'tK0|rBs', 'tK0|rK0', 'tK0|rK0|rF0',
    'tK0|rK0|rK0', 'tK0|tBm', 'tK0|tBm|tF0', 'tK0|tBp', 'tK0|tBp|tF0', 'tK0|tF0',
    'tK0|tK0', 'tK0|tK0|rBm', 'tK0|tK0|rBp', 'tK0|tK0|rK0', 'tRp', 'tRp|tCp',
    'tRp|tF0',
)


def sanitize_ngram_token(token: str) -> str:
    return token.replace("|", "__").replace("?", "Q")


def hand_ngram_doc(hand: dict[str, Any]) -> Counter:
    """Bag-of-ngrams document for a single sanitized hand."""
    actions = hand.get("actions") or []
    metadata = hand.get("metadata") or {}
    button_seat = metadata.get("button_seat")
    max_seats = metadata.get("max_seats") or 6

    tokens: list[str] = []
    grams: Counter = Counter()
    acting_seats: set = set()
    for action in actions:
        street = (action.get("street") or "x")[:1]
        act = NGRAM_ACTION_CODES.get(action.get("action_type") or "x", "X")
        amount = _safe_float(action.get("amount"), 0.0)
        pot_before = _safe_float(action.get("pot_before"), 0.0)
        if amount <= 0:
            bucket = "0"
        elif pot_before <= 0:
            bucket = "?"
        else:
            ratio = amount / pot_before
            bucket = "s" if ratio < 0.4 else ("m" if ratio < 0.9 else ("p" if ratio < 1.5 else "o"))
        token = street + act + bucket
        tokens.append(token)
        grams[token] += 1
        try:
            rel = (int(action.get("actor_seat")) - int(button_seat)) % int(max_seats)
            grams["pos" + str(rel) + act] += 1
        except Exception:
            pass
        acting_seats.add(action.get("actor_seat"))
    for i in range(len(tokens) - 1):
        grams[tokens[i] + "|" + tokens[i + 1]] += 1
        if i + 2 < len(tokens):
            grams[tokens[i] + "|" + tokens[i + 1] + "|" + tokens[i + 2]] += 1
    grams["len"] = len(tokens)
    grams["nseats"] = len(acting_seats)
    return grams



NGRAM_FEATURE_NAMES: list[str] = [
    "schema_ngram_" + sanitize_ngram_token(token) for token in NGRAM_VOCAB
]


def chunk_ngram_features(hands: Sequence[Mapping[str, Any]] | None) -> dict[str, float]:
    """Per-hand-normalized n-gram rates for one chunk."""
    chunk = list(hands or [])
    n = float(len(chunk))
    if n <= 0:
        return {name: 0.0 for name in NGRAM_FEATURE_NAMES}
    totals: Counter = Counter()
    for hand in chunk:
        if isinstance(hand, Mapping):
            totals.update(hand_ngram_doc(dict(hand)))
    return {
        "schema_ngram_" + sanitize_ngram_token(token): _safe_div(float(totals.get(token, 0.0)), n)
        for token in NGRAM_VOCAB
    }
