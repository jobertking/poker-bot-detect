"""Optional disk logging of validator DetectionSynapse requests.

Designed to stay off the hot path: serialize lightly on the caller, write in a
background thread. Enable with POKER44_LOG_REQUESTS=1.
"""

from __future__ import annotations

import gzip
import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class RequestLogger:
    """Queue-backed writer for miner request/response snapshots."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        log_dir: str | Path | None = None,
        full_payload: bool | None = None,
        gzip_files: bool | None = None,
        max_files: int | None = None,
        queue_size: int = 64,
    ):
        self.enabled = (
            bool(enabled) if enabled is not None else _env_bool("POKER44_LOG_REQUESTS", False)
        )
        self.log_dir = Path(
            log_dir
            or os.getenv("POKER44_REQUEST_LOG_DIR")
            or Path(__file__).resolve().parents[1] / "logs" / "requests"
        )
        self.full_payload = (
            bool(full_payload)
            if full_payload is not None
            else _env_bool("POKER44_LOG_REQUEST_FULL", True)
        )
        self.gzip_files = (
            bool(gzip_files)
            if gzip_files is not None
            else _env_bool("POKER44_LOG_REQUEST_GZIP", False)
        )
        # 0 / negative => keep all files (no rotation).
        if max_files is not None:
            self.max_files = int(max_files)
        else:
            self.max_files = _env_int("POKER44_REQUEST_LOG_MAX_FILES", 0)
        self._queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue(maxsize=queue_size)
        self._worker: Optional[threading.Thread] = None
        self._dropped = 0
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._worker = threading.Thread(
                target=self._run,
                name="poker44-request-logger",
                daemon=True,
            )
            self._worker.start()

    @property
    def dropped(self) -> int:
        return self._dropped

    def log(
        self,
        *,
        chunks: Sequence[Sequence[Mapping[str, Any]]] | None,
        risk_scores: Sequence[float] | None = None,
        predictions: Sequence[bool] | None = None,
        validator_hotkey: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        chunks_list = [list(chunk or []) for chunk in (chunks or [])]
        record: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "validator_hotkey": validator_hotkey or "",
            "n_chunks": len(chunks_list),
            "chunk_sizes": [len(chunk) for chunk in chunks_list],
            "n_hands": int(sum(len(chunk) for chunk in chunks_list)),
            "risk_scores": [float(s) for s in (risk_scores or [])],
            "predictions": [bool(p) for p in (predictions or [])],
        }
        if extra:
            record["extra"] = dict(extra)
        if self.full_payload:
            # Keep payload construction on the request thread; disk I/O is async.
            record["chunks"] = chunks_list

        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def close(self) -> None:
        if not self.enabled or self._worker is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._write(item)
                if self.max_files > 0:
                    self._prune()
            except Exception:
                # Never let logging crash the miner process.
                pass

    def _write(self, record: Mapping[str, Any]) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        hotkey = str(record.get("validator_hotkey") or "unknown")[:16]
        n_chunks = int(record.get("n_chunks") or 0)
        suffix = ".json.gz" if self.gzip_files else ".json"
        path = self.log_dir / f"{stamp}_{hotkey}_{n_chunks}c{suffix}"
        payload = json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if self.gzip_files:
            with gzip.open(path, "wb", compresslevel=3) as handle:
                handle.write(payload)
        else:
            path.write_bytes(payload)

    def _prune(self) -> None:
        patterns = ("*.json.gz", "*.json")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(self.log_dir.glob(pattern))
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[self.max_files :]:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass
