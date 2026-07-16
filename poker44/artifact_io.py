"""Helpers for atomic model writes, recipe fingerprints, and deploy locks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files that change trained artifact semantics. Bust recipe fingerprint when edited.
RECIPE_FILES = (
    "scripts/train/train_competitive_daily.py",
    "scripts/train/train_competitive_v3.py",
    "scripts/train/train_beat_v3_coherent.py",
    "scripts/train/train_rank_ensemble.py",
    "features/beat_v3_schema.py",
    "features/beat_v3_coherent_schema.py",
    "features/coherent_features.py",
    "features/competitive_fn_schema.py",
    "features/fn_patches.py",
    "features/competitive_schema.py",
    "features/competitive_features.py",
    "poker44/batch_calibration.py",
    "poker44/large_chunk_augment.py",
    "poker44/miner_inference.py",
    "poker44/validator/payload_view.py",
)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def recipe_fingerprint(repo_root: Path | None = None) -> str:
    root = Path(repo_root or REPO_ROOT)
    digest = hashlib.sha256()
    for relative in RECIPE_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(sha256_file(path).encode("ascii"))
        else:
            digest.update(b"missing")
        digest.update(b"\n")
    return digest.hexdigest()


def examples_fingerprint(path: Path) -> str:
    """Content fingerprint for labeled examples cache."""
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    digest.update(str(path.stat().st_size).encode("ascii"))
    digest.update(b":")
    digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def atomic_joblib_dump(obj: Any, path: Path | str) -> Path:
    """Write joblib via temp file + os.replace so readers never see a partial dump."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: Path | str, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def prune_archive(archive_dir: Path, *, keep: int = 30, patterns: Iterable[str] = ("current_*.joblib",)) -> int:
    """Keep newest `keep` archive files per pattern. keep<=0 => no prune."""
    if keep <= 0 or not archive_dir.is_dir():
        return 0
    removed = 0
    for pattern in patterns:
        files = sorted(archive_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            try:
                stale.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    return removed


class DeployLock:
    """Exclusive flock for daily refresh / concurrent trainers."""

    def __init__(self, lock_path: Path | str):
        self.lock_path = Path(lock_path)
        self._fh: Optional[Any] = None

    def acquire(self, *, blocking: bool = False) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.lock_path.open("a+", encoding="utf-8")
        try:
            import fcntl

            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            fcntl.flock(self._fh.fileno(), flags)
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.write(f"pid={os.getpid()}\n")
            self._fh.flush()
            return True
        except OSError:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            return False

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._fh.close()
        finally:
            self._fh = None

    def __enter__(self) -> "DeployLock":
        if not self.acquire(blocking=True):
            raise RuntimeError(f"Could not acquire deploy lock {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
