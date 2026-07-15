#!/usr/bin/env python3
"""Download Poker44 public benchmark releases with labels for local ML training.

Each training example is one chunk-group paired with its groundTruth label
(1=bot, 0=human). Labels stay index-aligned with wrapper['chunks'].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_URL = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "benchmark"


def http_get_json(url: str, *, timeout: float = 120.0, retries: int = 5) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if not payload.get("success", True):
                raise RuntimeError(f"API returned success=false for {url}: {payload}")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_err = exc
            sleep_s = min(2 ** attempt, 30)
            print(f"  retry {attempt}/{retries} after error: {exc} (sleep {sleep_s}s)", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed GET {url}: {last_err}")


def fetch_all_releases(base_url: str) -> list[dict[str, Any]]:
    releases: list[dict[str, Any]] = []
    before: str | None = None
    while True:
        params: dict[str, str] = {"limit": "100"}
        if before:
            params["before"] = before
        url = f"{base_url}/releases?{urllib.parse.urlencode(params)}"
        data = http_get_json(url)["data"]
        batch = data.get("releases") or []
        if not batch:
            break
        releases.extend(batch)
        next_cursor = data.get("nextCursor")
        if not next_cursor:
            # Fall back to paging by oldest sourceDate we have.
            oldest = min(r["sourceDate"] for r in batch)
            if before == oldest:
                break
            before = oldest
            # If API uses `before` as exclusive cursor by date, continue; else stop when no growth.
            if len(batch) < 100:
                break
        else:
            before = next_cursor
    # De-dupe by sourceDate, newest first.
    by_date: dict[str, dict[str, Any]] = {}
    for rel in releases:
        by_date[rel["sourceDate"]] = rel
    return sorted(by_date.values(), key=lambda r: r["sourceDate"])


def fetch_chunks_for_date(base_url: str, source_date: str, limit: int = 24) -> list[dict[str, Any]]:
    wrappers: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, str] = {"sourceDate": source_date, "limit": str(limit)}
        if cursor:
            params["cursor"] = cursor
        url = f"{base_url}/chunks?{urllib.parse.urlencode(params)}"
        data = http_get_json(url)["data"]
        batch = data.get("chunks") or []
        wrappers.extend(batch)
        cursor = data.get("nextCursor")
        if not cursor:
            break
    return wrappers


def flatten_labeled_examples(wrapper: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one API wrapper into labeled training rows (index-aligned)."""
    groups = wrapper.get("chunks") or []
    labels = wrapper.get("groundTruth")
    label_names = wrapper.get("groundTruthLabels")

    if labels is None:
        raise ValueError(
            f"Missing groundTruth for chunkId={wrapper.get('chunkId')} "
            f"sourceDate={wrapper.get('sourceDate')}"
        )
    if len(groups) != len(labels):
        raise ValueError(
            f"Label/group length mismatch for chunkId={wrapper.get('chunkId')}: "
            f"len(chunks)={len(groups)} len(groundTruth)={len(labels)}"
        )
    if label_names is not None and len(label_names) != len(labels):
        raise ValueError(
            f"groundTruthLabels length mismatch for chunkId={wrapper.get('chunkId')}: "
            f"len(groundTruth)={len(labels)} len(groundTruthLabels)={len(label_names)}"
        )

    examples: list[dict[str, Any]] = []
    for i, (group, y) in enumerate(zip(groups, labels)):
        y_int = int(y)
        if y_int not in (0, 1):
            raise ValueError(f"Unexpected label {y!r} at index {i} in {wrapper.get('chunkId')}")

        name = None
        if label_names is not None:
            name = label_names[i]
            expected = "bot" if y_int == 1 else "human"
            if str(name).lower() != expected:
                raise ValueError(
                    f"Label name mismatch at index {i} in {wrapper.get('chunkId')}: "
                    f"groundTruth={y_int} groundTruthLabels={name!r} expected={expected!r}"
                )

        examples.append(
            {
                "example_id": f"{wrapper.get('sourceDate')}:{wrapper.get('chunkHash')}:{i}",
                "sourceDate": wrapper.get("sourceDate"),
                "releaseVersion": wrapper.get("releaseVersion"),
                "schemaVersion": wrapper.get("schemaVersion")
                or wrapper.get("metadata", {}).get("schemaVersion"),
                "split": wrapper.get("split"),
                "chunkId": wrapper.get("chunkId"),
                "chunkHash": wrapper.get("chunkHash"),
                "chunkIndex": wrapper.get("chunkIndex"),
                "groupIndex": i,
                "label": y_int,  # 1=bot, 0=human
                "label_name": name if name is not None else ("bot" if y_int == 1 else "human"),
                "hand_count": len(group) if isinstance(group, list) else None,
                "hands": group,  # miner-visible hand payloads only
            }
        )
    return examples


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--limit", type=int, default=24, help="Page size for /chunks")
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Do not write per-wrapper raw JSON (examples + summary only)",
    )
    args = parser.parse_args()

    out: Path = args.out
    raw_dir = out / "raw"
    examples_path = out / "examples" / "examples.jsonl"
    summary_path = out / "manifest.json"

    out.mkdir(parents=True, exist_ok=True)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    if not args.skip_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching status from {args.base_url} ...", flush=True)
    status = http_get_json(args.base_url)["data"]
    write_json(out / "meta" / "status.json", status)
    print(
        f"  latestSourceDate={status.get('latestSourceDate')} "
        f"totalChunks={status.get('totalChunks')} totalHands={status.get('totalHands')}",
        flush=True,
    )

    print("Fetching releases ...", flush=True)
    releases = fetch_all_releases(args.base_url)
    write_json(out / "meta" / "releases.json", {"releases": releases})
    print(f"  {len(releases)} release dates: {releases[0]['sourceDate']} .. {releases[-1]['sourceDate']}", flush=True)

    label_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    total_wrappers = 0
    total_examples = 0
    total_hands = 0
    mismatches = 0

    if not releases:
        print("ERROR: no releases returned from API; keeping existing examples cache", flush=True)
        return 1

    tmp_examples = examples_path.with_suffix(".jsonl.tmp")
    try:
        with tmp_examples.open("w", encoding="utf-8") as examples_fp:
            for idx, rel in enumerate(releases, start=1):
                source_date = rel["sourceDate"]
                print(f"[{idx}/{len(releases)}] {source_date} ...", flush=True)
                wrappers = fetch_chunks_for_date(args.base_url, source_date, limit=args.limit)
                print(f"  wrappers={len(wrappers)} (release chunkCount={rel.get('chunkCount')})", flush=True)

                date_dir = raw_dir / source_date
                if not args.skip_raw:
                    date_dir.mkdir(parents=True, exist_ok=True)

                for wrapper in wrappers:
                    total_wrappers += 1
                    chunk_hash = wrapper.get("chunkHash") or wrapper.get("chunkId") or f"idx{total_wrappers}"
                    try:
                        examples = flatten_labeled_examples(wrapper)
                    except ValueError as exc:
                        mismatches += 1
                        print(f"  ERROR: {exc}", flush=True)
                        raise

                    if not args.skip_raw:
                        write_json(date_dir / f"{chunk_hash}.json", wrapper)

                    for ex in examples:
                        examples_fp.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        label_counts[ex["label"]] += 1
                        split_counts[str(ex.get("split") or "unknown")] += 1
                        date_counts[source_date] += 1
                        total_examples += 1
                        total_hands += int(ex.get("hand_count") or 0)

        if total_examples <= 0:
            print("ERROR: fetch produced 0 examples; keeping existing cache", flush=True)
            tmp_examples.unlink(missing_ok=True)
            return 1

        examples_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_examples, examples_path)
    except Exception:
        tmp_examples.unlink(missing_ok=True)
        raise

    summary = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "status": {
            "latestSourceDate": status.get("latestSourceDate"),
            "totalChunks": status.get("totalChunks"),
            "totalHands": status.get("totalHands"),
            "releaseVersion": status.get("releaseVersion"),
            "schemaVersion": status.get("schemaVersion"),
        },
        "releases": len(releases),
        "first_sourceDate": releases[0]["sourceDate"] if releases else None,
        "last_sourceDate": releases[-1]["sourceDate"] if releases else None,
        "wrappers": total_wrappers,
        "examples": total_examples,
        "hands": total_hands,
        "label_counts": {"human_0": label_counts.get(0, 0), "bot_1": label_counts.get(1, 0)},
        "split_counts": dict(split_counts),
        "examples_per_sourceDate": dict(sorted(date_counts.items())),
        "label_mismatches": mismatches,
        "paths": {
            "examples_jsonl": str(examples_path),
            "raw_dir": str(raw_dir) if not args.skip_raw else None,
            "manifest": str(summary_path),
        },
        "label_contract": {
            "label": "1=bot, 0=human",
            "alignment": "examples[i] from zip(wrapper['chunks'], wrapper['groundTruth'])",
            "note": "Never use hand_id/chunkId/dates/hashes as ML features",
        },
    }
    write_json(summary_path, summary)

    print("\n=== DONE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if mismatches:
        print(f"FAILED with {mismatches} label mismatches", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
