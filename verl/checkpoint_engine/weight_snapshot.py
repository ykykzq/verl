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
import math
import os
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass

import ray
import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


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
    max_bucket_bytes: int = 0
    # (bucket index, original name, original shape, element offset).
    chunks: tuple[tuple[int, str, tuple[int, ...], int], ...] = ()

    def __post_init__(self) -> None:
        _validate_integer("version", self.version)
        _validate_integer("tensor_count", self.tensor_count, minimum=1)
        _validate_integer("total_bytes", self.total_bytes)
        _validate_integer("max_bucket_bytes", self.max_bucket_bytes)
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
    weights: Iterable[tuple[str, torch.Tensor]],
    version: int,
    bucket_size: int,
    is_master: bool,
    *,
    split_tensors: bool = False,
) -> WeightSnapshot | None:
    """Collectively exhaust weights and publish detached CPU copies on rank zero.

    Buckets contain whole named tensors and are at most ``bucket_size`` bytes,
    except an individual oversized tensor, which occupies its own bucket unless
    ``split_tensors`` is enabled. RDMA exports split these tensors and account
    for the relay's 16-byte alignment between entries.
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

        started = time.perf_counter()
        serialization_s = copy_s = 0.0
        refs = []
        chunks = []
        max_bucket_bytes = 0

        def publish(bucket, size):
            nonlocal serialization_s, max_bucket_bytes
            tick = time.perf_counter()
            refs.append(ray.put(tuple(bucket)))
            serialization_s += time.perf_counter() - tick
            max_bucket_bytes = max(max_bucket_bytes, max(1, size))

        bucket = []
        bucket_bytes = tensor_count = total_bytes = 0
        names = set()
        for name, tensor in iterator:
            _validate_named_tensor(name, tensor, names)
            size = tensor.numel() * tensor.element_size()
            offset = (bucket_bytes + 15) // 16 * 16 if split_tensors else bucket_bytes
            if bucket and offset + size > bucket_size:
                publish(bucket, bucket_bytes)
                bucket = []
                bucket_bytes = 0
            # copy=True matters for already-CPU tensors; detach().cpu() alone
            # retains live model storage. The copy completes before advancing
            # generators whose context may restore merged LoRA parameters.
            tick = time.perf_counter()
            cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
            copy_s += time.perf_counter() - tick
            tensor_count += 1
            total_bytes += size
            if split_tensors and size > bucket_size:
                elements = bucket_size // tensor.element_size()
                if elements == 0:
                    raise ValueError("RDMA bucket cannot fit one tensor element")
                flat = cpu_tensor.reshape(-1)
                for start in range(0, flat.numel(), elements):
                    # Compact storage is necessary: Ray otherwise serializes the full tensor.
                    part = flat[start : start + elements].clone()
                    chunks.append((len(refs), name, tuple(tensor.shape), start))
                    publish([(name, part)], part.nbytes)
                continue
            if split_tensors:
                bucket_bytes = (bucket_bytes + 15) // 16 * 16
            bucket.append((name, cpu_tensor))
            bucket_bytes += size
            if bucket_bytes >= bucket_size:
                publish(bucket, bucket_bytes)
                bucket = []
                bucket_bytes = 0
        if bucket:
            publish(bucket, bucket_bytes)
        if not refs:
            raise ValueError("Cannot export an empty weight snapshot")
        logger.info(
            "weight_snapshot version=%s export_s=%.6f cpu_copy_s=%.6f serialization_s=%.6f "
            "bytes=%s max_bucket_bytes=%s buckets=%s",
            version,
            time.perf_counter() - started,
            copy_s,
            serialization_s,
            total_bytes,
            max_bucket_bytes,
            len(refs),
        )
        return WeightSnapshot(version, tuple(refs), tensor_count, total_bytes, max_bucket_bytes, tuple(chunks))
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
    chunks = {entry[0]: entry[1:] for entry in snapshot.chunks}
    if len(chunks) != len(snapshot.chunks) or any(index < 0 or index >= len(snapshot.buckets) for index in chunks):
        raise ValueError("Invalid or incomplete snapshot chunk metadata")
    pending = None
    offset = 0
    for index, ref in enumerate(snapshot.buckets):
        bucket = await ref
        if not isinstance(bucket, tuple | list) or not bucket:
            raise ValueError("Snapshot buckets must contain nonempty named-tensor sequences")
        if index in chunks:
            name, shape, expected_offset = chunks[index]
            if len(bucket) != 1 or bucket[0][0] != name:
                raise ValueError("Snapshot chunk identity mismatch")
            tensor = bucket[0][1]
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.ndim != 1:
                raise ValueError("Snapshot chunks must be flat CPU tensors")
            if pending is None:
                _validate_named_tensor(name, tensor, names)
                if expected_offset != 0 or any(type(dim) is not int or dim < 0 for dim in shape):
                    raise ValueError("Invalid first snapshot chunk")
                if math.prod(shape) * tensor.element_size() > snapshot.total_bytes - total_bytes:
                    raise ValueError("Snapshot chunk shape exceeds manifest totals")
                pending = (name, shape, torch.empty(math.prod(shape), dtype=tensor.dtype))
                offset = 0
            original_name, original_shape, merged = pending
            if (
                name != original_name
                or shape != original_shape
                or tensor.dtype != merged.dtype
                or offset != expected_offset
                or tensor.numel() == 0
                or offset + tensor.numel() > merged.numel()
            ):
                raise ValueError("Snapshot chunk offset, shape, or dtype mismatch")
            merged[offset : offset + tensor.numel()].copy_(tensor)
            offset += tensor.numel()
            total_bytes += tensor.nbytes
            if total_bytes > snapshot.total_bytes:
                raise ValueError("Snapshot chunks exceed manifest totals")
            if offset == merged.numel():
                tensor_count += 1
                pending = None
                yield name, merged.reshape(shape)
            continue
        if pending is not None:
            raise ValueError("Snapshot tensor is incomplete")
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
    if pending is not None:
        raise ValueError("Snapshot tensor is incomplete")
    if tensor_count != snapshot.tensor_count or total_bytes != snapshot.total_bytes:
        raise ValueError("Snapshot bucket contents do not match manifest totals")
