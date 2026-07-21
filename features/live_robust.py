"""Live-OOD feature selection for beat_v3_coherent.

Step 2 of the live-geometry pipeline: drop chunk-size-sensitive and
benchmark→live fragile columns so trees generalize at 70–160 hand chunks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Fragile after prepare_hand_for_miner / absolute BB scale shift (see top-miner audit).
_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "button_action_share",
    "hero_button_same",
    "_bb",
)

# Always drop raw chunk-size counters (direct hand-count leakage).
_EXCLUDE_EXACT: frozenset[str] = frozenset(
    {
        "hand_count",
        "fn_hand_count",
    }
)

# Order statistics that drift when live chunks are 72–160 vs benchmark 30–40.
_CHUNK_SIZE_ORDER_STATS: tuple[str, ...] = (
    "_min",
    "_max",
    "_q10",
    "_q90",
)

# Aggregate prefixes whose min/max/quantiles track N hands in chunk.
_SIZE_SENSITIVE_PREFIXES: tuple[str, ...] = (
    "schema_action_count_",
    "schema_street_count_",
    "schema_actor_",
    "schema_action_entropy_",
    "schema_actor_entropy_",
    "schema_street_entropy_",
    "schema_passive_share_",
    "schema_aggressive_share_",
    "schema_call_share_",
    "schema_check_share_",
    "schema_fold_share_",
    "schema_player_count_",
)


def is_robust_feature_name(name: str) -> bool:
    lowered = str(name).strip().lower()
    if not lowered:
        return False
    if lowered in _EXCLUDE_EXACT:
        return False
    if any(token in lowered for token in _EXCLUDE_SUBSTRINGS):
        return False
    for prefix in _SIZE_SENSITIVE_PREFIXES:
        if lowered.startswith(prefix):
            if any(lowered.endswith(suffix) for suffix in _CHUNK_SIZE_ORDER_STATS):
                return False
    return True


def filter_feature_names(names: Sequence[str]) -> list[str]:
    return [n for n in names if is_robust_feature_name(n)]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 8 or y.size < 8:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def audit_feature_hand_count_correlation(
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    hand_counts: Sequence[int],
    *,
    corr_threshold: float = 0.70,
) -> dict[str, Any]:
    """Flag features whose values track chunk hand count on live-shaped data."""
    hc = np.asarray(hand_counts, dtype=np.float64)
    flagged: list[dict[str, Any]] = []
    for j, name in enumerate(feature_names):
        col = feature_matrix[:, j]
        corr = abs(_pearson(col, hc))
        if corr >= corr_threshold:
            flagged.append({"name": name, "abs_corr_hand_count": corr})
    flagged.sort(key=lambda item: item["abs_corr_hand_count"], reverse=True)
    return {
        "corr_threshold": corr_threshold,
        "n_features": len(feature_names),
        "n_flagged": len(flagged),
        "flagged": flagged,
    }


def select_robust_features(
    all_names: Sequence[str],
    *,
    audit_report: Mapping[str, Any] | None = None,
) -> list[str]:
    """Rule-based drop + optional audit-driven drop (high |corr| with hand_count)."""
    kept = filter_feature_names(all_names)
    if audit_report:
        drop_audit = {
            item["name"]
            for item in (audit_report.get("flagged") or [])
            if item.get("name")
        }
        kept = [n for n in kept if n not in drop_audit]
    return kept


def load_request_log_chunks(
    log_dir: Path, *, max_files: int = 40, max_chunks_per_file: int = 8
) -> list[list[dict]]:
    """Load validator request chunks from disk logs (inputs only)."""
    files = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    chunks: list[list[dict]] = []
    for path in files[:max_files]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        batch = record.get("chunks") or []
        for chunk in batch[:max_chunks_per_file]:
            if isinstance(chunk, list) and chunk:
                chunks.append([h for h in chunk if isinstance(h, dict)])
    return chunks


def build_audit_report_from_logs(
    log_dir: Path,
    *,
    schema_module: Any,
    max_files: int = 40,
    corr_threshold: float = 0.70,
) -> dict[str, Any]:
    """Audit live request logs: which features correlate with hand count."""
    chunks = load_request_log_chunks(log_dir, max_files=max_files)
    if not chunks:
        return {"error": "no_chunks", "n_chunks": 0}

    names = list(schema_module.FEATURE_NAMES)
    rows = []
    hand_counts = []
    for hands in chunks:
        feat = schema_module.extract_chunk_features(hands)
        rows.append([float(feat.get(n, 0.0)) for n in names])
        hand_counts.append(len(hands))
    matrix = np.asarray(rows, dtype=np.float64)
    report = audit_feature_hand_count_correlation(
        matrix, names, hand_counts, corr_threshold=corr_threshold
    )
    report["n_chunks"] = len(chunks)
    report["hand_count_min"] = int(min(hand_counts))
    report["hand_count_max"] = int(max(hand_counts))
    report["hand_count_median"] = float(np.median(hand_counts))
    report["log_dir"] = str(log_dir)
    return report
