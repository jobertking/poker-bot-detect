#!/usr/bin/env python3
"""Build a labeled feature matrix from cached benchmark examples.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.chunk_features import FEATURE_NAMES, extract_chunk_features, features_to_vector

DEFAULT_EXAMPLES = ROOT / "data" / "benchmark" / "examples" / "examples.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "benchmark" / "features"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.examples.exists():
        raise SystemExit(f"Missing examples file: {args.examples}")

    meta_rows: list[dict] = []
    vectors: list[list[float]] = []

    with args.examples.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            ex = json.loads(line)
            feat = extract_chunk_features(ex.get("hands") or [])
            vec = features_to_vector(feat)
            if len(vec) != len(FEATURE_NAMES):
                raise RuntimeError(
                    f"Feature length mismatch at line {line_no}: "
                    f"{len(vec)} != {len(FEATURE_NAMES)}"
                )
            if any(v != v or v in (float("inf"), float("-inf")) for v in vec):
                raise RuntimeError(f"Non-finite feature at line {line_no}")

            meta_rows.append(
                {
                    "example_id": ex["example_id"],
                    "sourceDate": ex.get("sourceDate"),
                    "split": ex.get("split"),
                    "chunkHash": ex.get("chunkHash"),
                    "groupIndex": ex.get("groupIndex"),
                    "label": int(ex["label"]),
                    "label_name": ex.get("label_name"),
                    "hand_count_meta": ex.get("hand_count"),
                }
            )
            vectors.append(vec)

    X = np.asarray(vectors, dtype=np.float64)
    y = np.asarray([row["label"] for row in meta_rows], dtype=np.int64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "features.npz"
    csv_path = args.out_dir / "features.csv"
    names_path = args.out_dir / "feature_names.json"
    summary_path = args.out_dir / "features_summary.json"
    meta_path = args.out_dir / "meta.jsonl"

    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        feature_names=np.asarray(FEATURE_NAMES),
        example_ids=np.asarray([r["example_id"] for r in meta_rows]),
        sourceDates=np.asarray([r["sourceDate"] for r in meta_rows]),
        splits=np.asarray([r["split"] for r in meta_rows]),
    )

    fieldnames = [
        "example_id",
        "sourceDate",
        "split",
        "chunkHash",
        "groupIndex",
        "label",
        "label_name",
        "hand_count_meta",
        *FEATURE_NAMES,
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
        writer.writeheader()
        for meta, vec in zip(meta_rows, vectors):
            row = dict(meta)
            for name, value in zip(FEATURE_NAMES, vec):
                row[name] = value
            writer.writerow(row)

    with meta_path.open("w", encoding="utf-8") as meta_fp:
        for row in meta_rows:
            meta_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    names_path.write_text(json.dumps(FEATURE_NAMES, indent=2) + "\n", encoding="utf-8")

    from collections import Counter

    summary = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_counts": {str(k): int(v) for k, v in sorted(Counter(y.tolist()).items())},
        "split_counts": dict(Counter(r["split"] for r in meta_rows)),
        "n_sourceDates": len({r["sourceDate"] for r in meta_rows}),
        "X_shape": list(X.shape),
        "X_min": float(X.min()),
        "X_max": float(X.max()),
        "X_finite": bool(np.isfinite(X).all()),
        "paths": {
            "npz": str(npz_path),
            "csv": str(csv_path),
            "feature_names": str(names_path),
            "meta_jsonl": str(meta_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
