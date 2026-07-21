#!/usr/bin/env python3
"""Fetch new Poker44 benchmark days and retrain competitive model if needed.

Retraining runs the GATED live-geometry trainer (train_live_geometry.py): a new
candidate is deployed to current.joblib ONLY if it beats the currently-live
model on a live-shaped (topk100+topk120) holdout by >= --min-gain, using a fair
pool-held-out deploy (no holdout-date refit).
Otherwise the live artifact is left untouched and the candidate is saved aside.

Exit codes:
  0 — success (deployed, held, or already up to date)
  2 — fetch/train failure / lock busy
  3 — remote newer than local when --fail-if-stale
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker44.artifact_io import (
    DeployLock,
    examples_fingerprint,
    read_json,
    recipe_fingerprint,
)
from scripts.data.fetch_benchmark import BASE_URL, http_get_json

DEFAULT_BENCH = ROOT / "data" / "benchmark"
DEFAULT_MODEL = ROOT / "models" / "competitive"
DEFAULT_LOG = ROOT / "logs" / "daily_refresh"
DEFAULT_PYTHON = ROOT / ".venv_ml" / "bin" / "python"
DEFAULT_LOCK = ROOT / "logs" / "daily_refresh" / "deploy.lock"


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
    text = json.dumps(payload, indent=2) + "\n"
    path.write_text(text)
    (log_dir / "latest.json").write_text(text)
    return path


def maybe_reload_miner() -> str | None:
    """Optional PM2 reload so processes without hot-reload pick up artifact (best-effort)."""
    name = os.getenv("POKER44_PM2_RELOAD", "").strip()
    if not name:
        return None
    try:
        proc = subprocess.run(
            ["pm2", "reload", name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return f"pm2_reload:{name}:rc={proc.returncode}"
    except Exception as exc:
        return f"pm2_reload_failed:{exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--force-retrain", action="store_true", help="Retrain even if data/recipe unchanged")
    ap.add_argument("--skip-fetch", action="store_true", help="Skip API fetch (train on local cache)")
    ap.add_argument(
        "--fail-if-stale",
        action="store_true",
        help="Exit 3 if remote latestSourceDate is newer than local cache",
    )
    ap.add_argument("--holdout-days", type=int, default=2)
    ap.add_argument("--recent-val-days", type=int, default=4)
    ap.add_argument(
        "--min-gain",
        type=float,
        default=0.002,
        help="Required live-shaped topk100 reward gain over current model to deploy.",
    )
    ap.add_argument(
        "--force-deploy",
        action="store_true",
        help="Deploy candidate even if it does not beat current (NOT recommended).",
    )
    ap.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK)
    args = ap.parse_args()

    lock = DeployLock(args.lock_path)
    if not lock.acquire(blocking=False):
        write_run_log(
            args.log_dir,
            {
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "ok": False,
                "error": f"deploy lock busy: {args.lock_path}",
            },
        )
        print(f"ERROR: another refresh holds {args.lock_path}", flush=True)
        return 2

    try:
        return _run_locked(args)
    finally:
        lock.release()


def _run_locked(args: argparse.Namespace) -> int:
    py = str(args.python if args.python.exists() else sys.executable)
    examples = args.bench_dir / "examples" / "examples.jsonl"
    started = datetime.now(timezone.utc).isoformat()
    local_before = local_latest_date(examples)
    n_before = local_example_count(examples)
    fp_before = examples_fingerprint(examples) if examples.exists() else ""
    recipe_now = recipe_fingerprint(ROOT)
    prev_report = read_json(args.model_dir / "train_report.json")
    recipe_prev = str(prev_report.get("recipe_fingerprint") or "")

    print(f"Local latest={local_before} n={n_before}", flush=True)
    print(f"Recipe fingerprint={recipe_now[:12]}… prev={recipe_prev[:12] or 'none'}…", flush=True)

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
    fp_after = examples_fingerprint(examples) if examples.exists() else ""
    data_changed = (fp_after != fp_before) or (local_after != local_before) or (n_after != n_before)
    recipe_changed = (not recipe_prev) or (recipe_prev != recipe_now)
    model_missing = not (args.model_dir / "current.joblib").exists()
    print(
        f"After fetch latest={local_after} n={n_after} "
        f"data_changed={data_changed} recipe_changed={recipe_changed}",
        flush=True,
    )

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

    should_train = (
        args.force_retrain or data_changed or recipe_changed or model_missing
    )
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
            "examples_fingerprint": fp_after,
            "recipe_fingerprint": recipe_now,
        }
        path = write_run_log(args.log_dir, payload)
        print(f"Up to date; no retrain. Log: {path}", flush=True)
        return 0

    if n_after <= 0:
        write_run_log(
            args.log_dir,
            {
                "started_at_utc": started,
                "ok": False,
                "error": "no labeled examples available for training",
                "local_after": local_after,
            },
        )
        return 2

    train_cmd = [
        py,
        str(ROOT / "scripts" / "train" / "train_live_geometry.py"),
        "--examples",
        str(examples),
        "--out-dir",
        str(ROOT / "models" / "staging_live_geometry"),
        "--deploy-to-competitive",
        "--holdout-days",
        str(args.holdout_days),
        "--recent-val-days",
        str(args.recent_val_days),
        "--min-gain",
        str(args.min_gain),
    ]
    if args.force_deploy:
        train_cmd.append("--force-deploy")
    rc = run(train_cmd, cwd=ROOT)

    # The gated trainer returns 0 for both DEPLOYED and HELD. On deploy it
    # rewrites models/competitive/train_report.json; on hold it only updates
    # staging estimation_report.json. Prefer competitive report when fresh,
    # else fall back to staging.
    fresh_report: dict = {}
    deployed = False
    if rc == 0:
        deploy_report = read_json(args.model_dir / "train_report.json")
        staging_report = read_json(
            ROOT / "models" / "staging_live_geometry" / "estimation_report.json"
        )
        cand_report = read_json(args.model_dir / "candidate_report.json")
        if str(deploy_report.get("trained_at_utc") or "") >= started:
            fresh_report = deploy_report
        elif str(staging_report.get("trained_at_utc") or "") >= started:
            fresh_report = staging_report
        elif str(cand_report.get("trained_at_utc") or "") >= started:
            fresh_report = cand_report
        deployed = bool((fresh_report.get("gate") or {}).get("deployed"))

    # Only bother PM2-reloading when a new artifact actually landed (the miner
    # hot-reloads current.joblib on its own regardless).
    reload_note = maybe_reload_miner() if (rc == 0 and deployed) else None
    gate = fresh_report.get("gate") or {}
    payload = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": rc == 0,
        "fetched": fetched,
        "retrained": rc == 0,
        "deployed": deployed,
        "train_exit": rc,
        "local_before": local_before,
        "local_after": local_after,
        "remote_latest": remote_latest,
        "n_examples_before": n_before,
        "n_examples_after": n_after,
        "data_changed": data_changed,
        "recipe_changed": recipe_changed,
        "examples_fingerprint": fp_after,
        "recipe_fingerprint": recipe_now,
        "metrics": fresh_report.get("metrics") if rc == 0 else None,
        "gate": gate or None,
        "latest_source_date": fresh_report.get("latest_source_date") or local_after,
        "model_dir": str(args.model_dir),
        "miner_reload": reload_note,
        "retrain_triggers": {
            "force": bool(args.force_retrain),
            "data_changed": data_changed,
            "recipe_changed": recipe_changed,
            "model_missing": model_missing,
        },
    }
    path = write_run_log(args.log_dir, payload)
    if rc != 0:
        print(f"Train failed. Log: {path}", flush=True)
        return 2
    print(f"Done. Log: {path}", flush=True)
    print(
        f"Gate: deployed={deployed} "
        f"candidate_topk100={gate.get('candidate_live_topk100')} "
        f"current_topk100={gate.get('current_live_topk100')} "
        f"gain={gate.get('gain')} min_gain={gate.get('min_gain')}",
        flush=True,
    )
    if reload_note:
        print(reload_note, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
