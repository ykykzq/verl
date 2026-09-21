# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pluggable trajectory x replica scheduling and migration gates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from verl.utils.import_utils import load_class_from_fqn


@dataclass(frozen=True)
class ReplicaSnapshot:
    server_id: str
    inflight: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MigrationContext:
    trajectory: dict[str, Any]
    source: ReplicaSnapshot
    target: ReplicaSnapshot


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str = ""


class TrajectoryReplicaScorer(Protocol):
    """Scores one trajectory x replica pair. Higher scores are preferred."""

    def score(self, trajectory: dict[str, Any], replica: ReplicaSnapshot) -> float: ...


class TrajectoryMigrationGate(Protocol):
    """Allows or rejects a proposed movement between two replicas."""

    def evaluate(self, context: MigrationContext) -> GateResult: ...


class LoadAwareTrajectoryReplicaScorer:
    """Combines current load with stable trajectory-to-replica affinity."""

    def __init__(self, load_weight: float = 1.0, affinity_weight: float = 0.001):
        self.load_weight = load_weight
        self.affinity_weight = affinity_weight

    def score(self, trajectory: dict[str, Any], replica: ReplicaSnapshot) -> float:
        request_id = str(trajectory["request_id"])
        digest = hashlib.sha256(f"{request_id}\0{replica.server_id}".encode()).digest()
        affinity = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return -self.load_weight * replica.inflight + self.affinity_weight * affinity


class DifferentReplicaGate:
    def evaluate(self, context: MigrationContext) -> GateResult:
        allowed = context.source.server_id != context.target.server_id
        return GateResult(allowed, "source and target are the same replica" if not allowed else "")


class SameModelGate:
    def evaluate(self, context: MigrationContext) -> GateResult:
        source_model = context.source.metadata.get("model_id")
        target_model = context.target.metadata.get("model_id")
        allowed = source_model is not None and source_model == target_model
        return GateResult(allowed, "source and target model identities differ or are unknown" if not allowed else "")


class SameWeightVersionGate:
    """Fail closed when either weight version is unavailable."""

    def evaluate(self, context: MigrationContext) -> GateResult:
        source_version = context.source.metadata.get("weight_version")
        target_version = context.target.metadata.get("weight_version")
        allowed = source_version is not None and source_version == target_version
        return GateResult(allowed, "source and target weight versions differ or are unknown" if not allowed else "")


class KVTransferCapabilityGate:
    def __init__(self, backend: str = "remote_prefix"):
        self.backend = backend

    def evaluate(self, context: MigrationContext) -> GateResult:
        source_backends = context.source.metadata.get("kv_transfer_backends", [])
        target_backends = context.target.metadata.get("kv_transfer_backends", [])
        source_domain = context.source.metadata.get("kv_transfer_domain")
        target_domain = context.target.metadata.get("kv_transfer_domain")
        source_namespace = context.source.metadata.get("kv_transfer_namespace")
        target_namespace = context.target.metadata.get("kv_transfer_namespace")
        available = self.backend in source_backends and self.backend in target_backends
        allowed = (
            available
            and source_domain is not None
            and source_domain == target_domain
            and source_namespace is not None
            and source_namespace == target_namespace
        )
        if not available:
            reason = f"KV transfer backend {self.backend!r} is not available on both replicas"
        elif source_domain is None or source_domain != target_domain:
            reason = "source and target KV transfer domains differ or are unknown"
        elif source_namespace is None or source_namespace != target_namespace:
            reason = "source and target KV transfer namespaces differ or are unknown"
        else:
            reason = ""
        return GateResult(allowed, reason)


DEFAULT_SCORER = "verl.workers.rollout.trajectory_scheduler.LoadAwareTrajectoryReplicaScorer"
DEFAULT_GATES = (
    "verl.workers.rollout.trajectory_scheduler.DifferentReplicaGate",
    "verl.workers.rollout.trajectory_scheduler.SameModelGate",
    "verl.workers.rollout.trajectory_scheduler.SameWeightVersionGate",
    "verl.workers.rollout.trajectory_scheduler.KVTransferCapabilityGate",
)


class TrajectoryScheduler:
    """Selects a target and applies an ordered, extensible gate chain."""

    def __init__(self, config: dict[str, Any]):
        scorer_class = config.get("scorer_class") or DEFAULT_SCORER
        scorer_type = load_class_from_fqn(scorer_class, "trajectory replica scorer")
        self.scorer: TrajectoryReplicaScorer = scorer_type(**config.get("scorer_kwargs", {}))

        gate_classes = config.get("gate_classes") or list(DEFAULT_GATES)
        gate_kwargs = config.get("gate_kwargs", {})
        self.gates: list[TrajectoryMigrationGate] = []
        for gate_class in gate_classes:
            gate_type = load_class_from_fqn(gate_class, "trajectory migration gate")
            kwargs = dict(gate_kwargs.get(gate_class, {}))
            if gate_class.endswith("KVTransferCapabilityGate"):
                kwargs.setdefault("backend", config.get("kv_transfer_backend", "remote_prefix"))
            self.gates.append(gate_type(**kwargs))
        self.min_score_improvement = float(config.get("min_score_improvement", 0.0))

    def choose(
        self,
        trajectory: dict[str, Any],
        source_server_id: str,
        replicas: list[ReplicaSnapshot],
    ) -> tuple[ReplicaSnapshot | None, dict[str, Any]]:
        by_id = {replica.server_id: replica for replica in replicas}
        source = by_id.get(source_server_id)
        if source is None:
            return None, {"reason": "source replica is no longer registered"}

        scores = {replica.server_id: self.scorer.score(trajectory, replica) for replica in replicas}
        candidates = sorted(
            (replica for replica in replicas if replica.server_id != source_server_id),
            key=lambda replica: (scores[replica.server_id], replica.server_id),
            reverse=True,
        )
        rejections = []
        for target in candidates:
            improvement = scores[target.server_id] - scores[source.server_id]
            if improvement < self.min_score_improvement:
                rejections.append(
                    {
                        "server_id": target.server_id,
                        "reason": "score improvement is below the configured threshold",
                    }
                )
                continue

            context = MigrationContext(trajectory=trajectory, source=source, target=target)
            for gate in self.gates:
                result = gate.evaluate(context)
                if not result.allowed:
                    rejections.append(
                        {
                            "server_id": target.server_id,
                            "reason": result.reason,
                            "rejected_by": f"{type(gate).__module__}.{type(gate).__qualname__}",
                        }
                    )
                    break
            else:
                return target, {
                    "scores": scores,
                    "score_improvement": improvement,
                    "rejected_candidates": rejections,
                }

        diagnostics = {
            "scores": scores,
            "reason": "no candidate replica passed the migration gates",
            "rejected_candidates": rejections,
        }
        if rejections:
            diagnostics.update({key: value for key, value in rejections[0].items() if key != "server_id"})
        return None, diagnostics
