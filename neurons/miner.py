"""Poker44 miner: competitive ensemble inference on DetectionSynapse requests."""

# from __future__ import annotations

import hashlib
import re
import subprocess
import time
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlsplit, urlunsplit

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner_inference import XgbBotRiskModel
from poker44.request_logger import RequestLogger
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

DEFAULT_PUBLIC_REPO_URL = "https://github.com/jobertking/poker-bot-detect"


class Miner(BaseMinerNeuron):
    """
    Production miner: competitive ensemble + calibration + batch top-K safety.

    Validators send DetectionSynapse(chunks=...); miner returns risk_scores.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]

        self.risk_model = XgbBotRiskModel()
        self._repo_root = repo_root
        bt.logging.info(
            f"Loaded competitive inference model | path={self.risk_model.path} "
            f"threshold={self.risk_model.threshold} "
            f"batch={self.risk_model.batch_mode} "
            f"n_features={len(self.risk_model.feature_names)}"
        )

        runtime_commit = self._repo_head(repo_root)
        runtime_repo_url = (
            self._normalize_repo_url(self._repo_url(repo_root)) or DEFAULT_PUBLIC_REPO_URL
        )
        artifact_path = Path(self.risk_model.path)
        artifact_sha256 = (
            self._sha256_file(artifact_path) if artifact_path.is_file() else ""
        )
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=self._implementation_files(repo_root),
            defaults={
                "model_name": self.risk_model.model_name,
                "model_version": self.risk_model.model_version,
                "framework": "beat-v3-xgb",
                "license": "MIT",
                "repo_url": runtime_repo_url,
                "repo_commit": runtime_commit,
                "artifact_url": str(artifact_path.resolve()) if artifact_path.is_file() else "",
                "artifact_sha256": artifact_sha256,
                "notes": (
                    "Beat-v3 miner: competitive+FN+v3 features, capacity XGB, "
                    "sanitize + LODO cal + top-K. Beats xgb_v3_holdout on sealed 7/13-14 reward. "
                    f"Artifact: {artifact_path}."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on Poker44 public benchmark API labeled chunk-groups "
                    "after prepare_hand_for_miner sanitization "
                    "(https://api.poker44.net/api/v1/benchmark). "
                    "Does not use validator-only live eval labels."
                ),
                "training_data_sources": [
                    "https://api.poker44.net/api/v1/benchmark"
                ],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self.request_logger = RequestLogger()
        if self.request_logger.enabled:
            bt.logging.info(
                "Validator request disk logging enabled | "
                f"dir={self.request_logger.log_dir} "
                f"full_payload={self.request_logger.full_payload} "
                f"gzip={self.request_logger.gzip_files} "
                f"max_files={self.request_logger.max_files}"
            )
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Axon created: {self.axon}")

    @classmethod
    def _implementation_files(cls, repo_root: Path) -> List[Path]:
        files = [Path(__file__).resolve()]
        for relative in (
            "poker44/miner_inference.py",
            "poker44/batch_calibration.py",
            "features/competitive_features.py",
            "features/competitive_schema.py",
            "features/fn_patches.py",
            "features/competitive_fn_schema.py",
            "features/beat_v3_schema.py",
            "features/beat_v3_coherent_schema.py",
            "features/beat_v3_coherent_live_schema.py",
            "features/coherent_features.py",
            "features/hand_ngram_features.py",
            "features/chunk_features.py",
            "features/merged_schema.py",
            "poker44/large_chunk_augment.py",
        ):
            candidate = repo_root / relative
            if candidate.exists():
                files.append(candidate)
        return files

    @staticmethod
    def _repo_head(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _repo_url(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _normalize_repo_url(url: str) -> str:
        cleaned = str(url or "").strip()
        if not cleaned:
            return ""
        if cleaned.startswith("git@"):
            host_path = cleaned.split(":", 1)
            if len(host_path) == 2:
                host = host_path[0][4:]
                path = host_path[1]
                if path.endswith(".git"):
                    path = path[:-4]
                return f"https://{host}/{path}"
        # Strip embedded credentials (user:token@host) before publishing.
        cleaned = re.sub(r"https?://[^/@]+@", "https://", cleaned)
        if cleaned.endswith(".git"):
            cleaned = cleaned[:-4]
        parts = urlsplit(cleaned)
        if parts.scheme and parts.netloc:
            cleaned = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        return cleaned.rstrip("/")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']} "
            f"policy_violations={self.manifest_compliance.get('policy_violations', [])})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Inference: one bot-risk score per received chunk-group."""
        if self.risk_model.maybe_reload():
            bt.logging.info(
                f"Hot-reloaded model | path={self.risk_model.path} "
                f"version={self.risk_model.model_version} "
                f"n_features={len(self.risk_model.feature_names)}"
            )
            self._refresh_artifact_manifest_fields()
        chunks = synapse.chunks or []
        scores, predictions = self.risk_model.predict(chunks)
        synapse.risk_scores = scores
        synapse.predictions = predictions
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"Inference scored {len(chunks)} chunks | "
            f"pred_pos={sum(1 for p in predictions if p)} "
            f"mean_score={sum(scores) / len(scores) if scores else 0.0:.4f}"
        )
        validator_hotkey = ""
        if synapse.dendrite is not None and getattr(synapse.dendrite, "hotkey", None):
            validator_hotkey = str(synapse.dendrite.hotkey)
        self.request_logger.log(
            chunks=chunks,
            risk_scores=scores,
            predictions=predictions,
            validator_hotkey=validator_hotkey,
            extra={"dropped_logs": self.request_logger.dropped},
        )
        return synapse

    def _refresh_artifact_manifest_fields(self) -> None:
        artifact_path = Path(self.risk_model.path)
        self.model_manifest["model_name"] = self.risk_model.model_name
        self.model_manifest["model_version"] = self.risk_model.model_version
        if artifact_path.is_file():
            self.model_manifest["artifact_url"] = str(artifact_path.resolve())
            self.model_manifest["artifact_sha256"] = self._sha256_file(artifact_path)
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)

    def score_chunk(self, chunk: list) -> float:
        """Sync helper used by local sims/tests."""
        return self.risk_model.score_chunk(chunk)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("XGBoost Poker44 miner running (inference mode)...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
