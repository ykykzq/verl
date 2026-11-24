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
"""Startup-only, bounded GPU roles for immutable weight publication."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SnapshotMemoryRole:
    converter: bool
    snapshot_bytes: int
    conversion_input_bytes: int
    conversion_workspace_bytes: int
    conversion_output_bytes: int
    installation_bytes: int
    communication_bytes: int

    @property
    def total_bytes(self):
        return sum(value for name, value in asdict(self).items() if name != "converter")

    def as_dict(self):
        return {**asdict(self), "total_bytes": self.total_bytes}


def snapshot_memory_role(config, replica_rank, bucket_bytes):
    """Freeze an explicitly selected converter before any engine allocates KV.

    Conversion workspace includes the new outputs while they are being built.
    The separate output budget bounds already completed, unacknowledged outputs.
    Input storage lives in the checkpoint worker, outside the converter budget.
    """
    mode = getattr(config, "memory_planning", "legacy")
    if mode not in ("legacy", "role"):
        raise ValueError("snapshot memory_planning must be legacy or role")
    if mode == "legacy":
        return SnapshotMemoryRole(
            True,
            config.max_bytes_per_gpu,
            0,
            config.max_conversion_bytes * (2 if getattr(config, "pipeline_depth", 0) else 1),
            0,
            config.max_tensor_bytes,
            bucket_bytes,
        )
    if config.pipeline_depth != 2:
        raise ValueError("role memory planning requires the bounded pipeline")
    selected = getattr(config, "converter_replica_rank", None)
    location = getattr(config, "conversion_location", "rollout")
    if location == "rollout" and (type(selected) is not int or selected < 0):
        raise ValueError("role planning requires a fixed converter_replica_rank at startup")
    converter = replica_rank == selected and location == "rollout"
    inputs = 2 * config.max_tensor_bytes
    outputs = config.max_conversion_bytes - inputs
    if outputs <= 0:
        raise ValueError("conversion budget cannot hold two inputs and retained outputs")
    return SnapshotMemoryRole(
        converter,
        config.max_bytes_per_gpu,
        inputs if converter else 0,
        config.max_conversion_bytes if converter else 0,
        outputs if converter else 0,
        config.max_tensor_bytes,
        bucket_bytes,
    )


def fixed_converter_workers(config, replica_ranks, tp_size):
    """Return indices into the sorted replica worker list, never reassign roles."""
    selected = getattr(config, "converter_replica_rank", None)
    if selected not in replica_ranks:
        raise ValueError(f"reserved converter replica {selected} is absent")
    base = replica_ranks.index(selected) * tp_size
    return list(range(base, base + tp_size))
