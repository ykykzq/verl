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

"""Immutable CPU weight snapshots with independently fetchable Ray buckets."""

import logging
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass

import ray
import torch

logger = logging.getLogger(__name__)


def _validate_integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


@dataclass(frozen=True)
class WeightSnapshot:
    """A complete version and nested refs that keep its CPU buckets alive."""

    version: int
    buckets: tuple[ray.ObjectRef, ...]
    tensor_count: int
    total_bytes: int

    def __post_init__(self) -> None:
        _validate_integer("version", self.version)
        _validate_integer("tensor_count", self.tensor_count, minimum=1)
        _validate_integer("total_bytes", self.total_bytes)
        if not isinstance(self.buckets, tuple):
            raise TypeError("Snapshot buckets must be an immutable tuple")
        if not self.buckets:
            raise ValueError("Snapshot must not be empty")
        if not all(isinstance(ref, ray.ObjectRef) for ref in self.buckets):
            raise TypeError("Snapshot buckets must contain Ray ObjectRefs")


def _drain_weights(weights: Iterator) -> None:
    """Finish outstanding collective generator entries after a local error."""
    try:
        for _ in weights:
            pass
    except Exception:
        # Preserve the original export error, including when the iterator itself
        # fails while draining. A broken collective cannot be repaired locally.
        logger.exception("Failed to drain collective weight export after a local error")


def _validate_named_tensor(name, tensor, names: set[str]) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Weight names must be nonempty strings")
    if name in names:
        raise ValueError(f"Duplicate weight name: {name}")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Weight {name!r} must be a torch.Tensor")
    names.add(name)


def export_weight_snapshot(
    weights: Iterable[tuple[str, torch.Tensor]], version: int, bucket_size: int, is_master: bool
) -> WeightSnapshot | None:
    """Collectively exhaust weights and publish detached CPU copies on rank zero.

    Buckets contain whole named tensors and are at most ``bucket_size`` bytes,
    except an individual oversized tensor, which occupies its own bucket.
    Every non-source rank consumes the same iterator without materialization.
    Local copy/put/validation errors drain the remaining collective entries
    before propagating; callers must also await all participating worker RPCs.
    """
    iterator = iter(weights)
    try:
        _validate_integer("version", version)
        _validate_integer("bucket_size", bucket_size, minimum=1)
        if not is_master:
            for _ in iterator:
                pass
            return None

        refs = []
        bucket = []
        bucket_bytes = tensor_count = total_bytes = 0
        names = set()
        for name, tensor in iterator:
            _validate_named_tensor(name, tensor, names)
            size = tensor.numel() * tensor.element_size()
            if bucket and bucket_bytes + size > bucket_size:
                refs.append(ray.put(tuple(bucket)))
                bucket = []
                bucket_bytes = 0
            # copy=True matters for already-CPU tensors; detach().cpu() alone
            # retains live model storage. The copy completes before advancing
            # generators whose context may restore merged LoRA parameters.
            cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
            bucket.append((name, cpu_tensor))
            bucket_bytes += size
            tensor_count += 1
            total_bytes += size
            if bucket_bytes >= bucket_size:
                refs.append(ray.put(tuple(bucket)))
                bucket = []
                bucket_bytes = 0
        if bucket:
            refs.append(ray.put(tuple(bucket)))
        if not refs:
            raise ValueError("Cannot export an empty weight snapshot")
        return WeightSnapshot(version, tuple(refs), tensor_count, total_bytes)
    except Exception:
        _drain_weights(iterator)
        raise


async def iter_snapshot_weights(snapshot: WeightSnapshot) -> AsyncIterator[tuple[str, torch.Tensor]]:
    """Fetch one nested CPU bucket at a time and yield consumer-owned tensors.

    Clones prevent an in-place adapter or failed installation from corrupting
    the retained snapshot used by other replicas and retries.
    """
    if not isinstance(snapshot, WeightSnapshot):
        raise TypeError("Expected a WeightSnapshot manifest")
    snapshot.__post_init__()
    tensor_count = total_bytes = 0
    names = set()
    for ref in snapshot.buckets:
        bucket = await ref
        if not isinstance(bucket, tuple | list) or not bucket:
            raise ValueError("Snapshot buckets must contain nonempty named-tensor sequences")
        for item in bucket:
            if not isinstance(item, tuple | list) or len(item) != 2:
                raise ValueError("Snapshot bucket entries must be (name, tensor) pairs")
            name, tensor = item
            _validate_named_tensor(name, tensor, names)
            if tensor.device.type != "cpu":
                raise ValueError(f"Snapshot weight {name!r} must be on CPU")
            tensor_count += 1
            total_bytes += tensor.numel() * tensor.element_size()
            if tensor_count > snapshot.tensor_count or total_bytes > snapshot.total_bytes:
                raise ValueError("Snapshot bucket contents exceed manifest totals")
            yield name, tensor.detach().clone(memory_format=torch.contiguous_format)
        del bucket
    if tensor_count != snapshot.tensor_count or total_bytes != snapshot.total_bytes:
        raise ValueError("Snapshot bucket contents do not match manifest totals")
