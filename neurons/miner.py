"""Poker44 miner: XGBoost inference on validator DetectionSynapse requests."""

# from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner_inference import XgbBotRiskModel
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Production-style miner: load trained XGBoost and run inference per chunk.

    Validators send DetectionSynapse(chunks=...); miner returns risk_scores.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]

        self.risk_model = XgbBotRiskModel()
        bt.logging.info(
            f"Loaded XGBoost inference model | dir={self.risk_model.model_dir} "
            f"threshold={self.risk_model.threshold} "
            f"n_features={len(self.risk_model.feature_names)}"
        )

        impl_files = [
            Path(__file__).resolve(),
            repo_root / "poker44" / "miner_inference.py",
            repo_root / "features" / "chunk_features.py",
        ]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=impl_files,
            defaults={
                "model_name": self.risk_model.model_name,
                "model_version": self.risk_model.model_version,
                "framework": "xgboost",
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": (
                    "XGBoost chunk-level bot-risk miner. "
                    f"Artifacts under {self.risk_model.model_dir}."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on Poker44 public benchmark API labeled chunk-groups "
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
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
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
        return synapse

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
