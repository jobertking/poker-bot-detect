#!/usr/bin/env python3
"""Fetch new Poker44 benchmark days and retrain competitive model if data changed.

Exit codes:
  0 — success (retrained, or already up to date with --no-retrain-if-fresh)
  2 — fetch/train failure
  3 — no new data and --fail-if-stale was set
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.fetch_benchmark import http_get_json, BASE_URL

DEFAULT_BENCH = ROOT / "data" / "benchmark"
DEFAULT_MODEL = ROOT / "models" / "competitive"
DEFAULT_LOG = ROOT / "logs" / "daily_refresh"
DEFAULT_PYTHON = ROOT / ".venv_ml" / "bin" / "python"


def local_latest_date(examples: Path) -> str | None:
    if not examples.exists():
        return None
    latest = None
    with examples.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line).get("sourceDate")
            if d and (latest is None or d > latest):
                latest = d
    return latest


def local_example_count(examples: Path) -> int:
    if not examples.exists():
        return 0
    n = 0
    with examples.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def run(cmd: list[str], *, cwd: Path) -> int:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def write_run_log(log_dir: Path, payload: dict) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"run_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    (log_dir / "latest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--force-retrain", action="store_true", help="Retrain even if no new dates")
    ap.add_argument("--skip-fetch", action="store_true", help="Skip API fetch (train on local cache)")
    ap.add_argument("--fail-if-stale", action="store_true", help="Exit 3 if remote has newer day and we skipped fetch")
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--recent-val-days", type=int, default=4)
    args = ap.parse_args()

    py = str(args.python if args.python.exists() else sys.executable)
    examples = args.bench_dir / "examples" / "examples.jsonl"
    started = datetime.now(timezone.utc).isoformat()
    local_before = local_latest_date(examples)
    n_before = local_example_count(examples)

    print(f"Local latest={local_before} n={n_before}", flush=True)

    remote_latest = None
    try:
        status = http_get_json(args.base_url)["data"]
        remote_latest = status.get("latestSourceDate")
        print(f"Remote latest={remote_latest}", flush=True)
    except Exception as exc:
        print(f"WARN: could not reach API status: {exc}", flush=True)
        if not args.skip_fetch and not examples.exists():
            write_run_log(
                args.log_dir,
                {
                    "started_at_utc": started,
                    "ok": False,
                    "error": f"API unreachable and no local cache: {exc}",
                },
            )
            return 2

    need_fetch = not args.skip_fetch
    # Fetch whenever possible so same-day chunk growth is picked up.
    fetched = False
    if need_fetch:
        rc = run(
            [
                py,
                str(ROOT / "scripts" / "data" / "fetch_benchmark.py"),
                "--out",
                str(args.bench_dir),
                "--base-url",
                args.base_url,
            ],
            cwd=ROOT,
        )
        if rc != 0:
            write_run_log(
                args.log_dir,
                {
                    "started_at_utc": started,
                    "ok": False,
                    "error": f"fetch_benchmark exit {rc}",
                    "local_before": local_before,
                    "remote_latest": remote_latest,
                },
            )
            return 2
        fetched = True
    else:
        print("Skip fetch (--skip-fetch)", flush=True)

    local_after = local_latest_date(examples)
    n_after = local_example_count(examples)
    data_changed = (local_after != local_before) or (n_after != n_before)
    print(f"After fetch latest={local_after} n={n_after} changed={data_changed}", flush=True)

    if (
        args.fail_if_stale
        and remote_latest
        and local_after
        and remote_latest > local_after
    ):
        write_run_log(
            args.log_dir,
            {
                "started_at_utc": started,
                "ok": False,
                "error": "stale vs remote",
                "local_after": local_after,
                "remote_latest": remote_latest,
            },
        )
        return 3

    should_train = args.force_retrain or data_changed or not (args.model_dir / "current.joblib").exists()
    if not should_train:
        payload = {
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "ok": True,
            "fetched": fetched,
            "retrained": False,
            "reason": "already_fresh",
            "local_latest": local_after,
            "remote_latest": remote_latest,
            "n_examples": n_after,
        }
        path = write_run_log(args.log_dir, payload)
        print(f"Up to date; no retrain. Log: {path}", flush=True)
        return 0

    rc = run(
        [
            py,
            str(ROOT / "scripts" / "train" / "train_competitive_daily.py"),
            "--examples",
            str(examples),
            "--out-dir",
            str(args.model_dir),
            "--holdout-days",
            str(args.holdout_days),
            "--recent-val-days",
            str(args.recent_val_days),
            "--archive",
        ],
        cwd=ROOT,
    )
    report = {}
    report_path = args.model_dir / "train_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    payload = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": rc == 0,
        "fetched": fetched,
        "retrained": rc == 0,
        "fetch_exit": 0 if fetched else None,
        "train_exit": rc,
        "local_before": local_before,
        "local_after": local_after,
        "remote_latest": remote_latest,
        "n_examples_before": n_before,
        "n_examples_after": n_after,
        "metrics": report.get("metrics"),
        "latest_source_date": report.get("latest_source_date") or local_after,
        "model_dir": str(args.model_dir),
    }
    path = write_run_log(args.log_dir, payload)
    if rc != 0:
        print(f"Train failed. Log: {path}", flush=True)
        return 2
    print(f"Done. Log: {path}", flush=True)
    if report.get("metrics"):
        sealed = report["metrics"].get("holdout_sealed") or {}
        print(
            f"Sealed reward={sealed.get('reward')} bot@5fpr={sealed.get('bot_recall_at_5fpr')} "
            f"ap={sealed.get('ap')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
