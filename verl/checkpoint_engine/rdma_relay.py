# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""CPU-only registered weight buckets for Mooncake RDMA relay chains."""

import asyncio
import copy
import hashlib
import logging
import math
import os
import time
from uuid import uuid4

import ray
import torch

_ALIGNMENT = 16
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    if hasattr(torch, name)
}


def _aligned(offset):
    return (offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _checksum(buffer):
    return hashlib.sha256(memoryview(buffer.numpy())).hexdigest()


def _layout(bucket):
    if not isinstance(bucket, (tuple, list)) or not bucket:
        raise ValueError("relay bucket must be a nonempty named-tensor sequence")
    tensors, names, end = [], set(), 0
    for item in bucket:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("relay bucket entries must be (name, tensor) pairs")
        name, tensor = item
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("relay tensor names must be nonempty and unique")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"relay weight {name!r} must be a tensor")
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("relay weights must be dense, non-quantized CPU tensors")
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported relay tensor dtype {dtype}")
        offset = _aligned(end)
        nbytes = tensor.numel() * tensor.element_size()
        tensors.append(dict(name=name, dtype=dtype, shape=list(tensor.shape), offset=offset, nbytes=nbytes))
        names.add(name)
        end = offset + nbytes
    return tensors, max(1, end)


def _validate_descriptor(descriptor):
    if not isinstance(descriptor, dict) or descriptor.get("format_version") != 1:
        raise ValueError("unsupported relay descriptor format")
    for key in ("bucket_id", "node_id", "session_id"):
        if not isinstance(descriptor.get(key), str) or not descriptor[key]:
            raise ValueError(f"invalid relay descriptor {key}")
    for key in ("pointer", "total_bytes"):
        if type(descriptor.get(key)) is not int or descriptor[key] <= 0:
            raise ValueError(f"invalid relay descriptor {key}")
    digest = descriptor.get("checksum")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid relay checksum")
    tensors = descriptor.get("tensors")
    if not isinstance(tensors, (list, tuple)) or not tensors:
        raise ValueError("invalid relay tensor layout")
    names, end = set(), 0
    for meta in tensors:
        if not isinstance(meta, dict):
            raise ValueError("invalid relay tensor layout entry")
        name, dtype, shape = meta.get("name"), meta.get("dtype"), meta.get("shape")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("invalid relay tensor layout name")
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise ValueError("invalid relay tensor layout dtype")
        if not isinstance(shape, (list, tuple)) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("invalid relay tensor layout shape")
        nbytes = math.prod(shape) * _DTYPES[dtype].itemsize
        if type(meta.get("nbytes")) is not int or meta["nbytes"] != nbytes:
            raise ValueError("invalid relay tensor layout byte count")
        if type(meta.get("offset")) is not int or meta["offset"] != _aligned(end):
            raise ValueError("invalid relay tensor layout offset")
        end = meta["offset"] + nbytes
        if end > descriptor["total_bytes"]:
            raise ValueError("relay tensor layout exceeds buffer bounds")
        names.add(name)
    if descriptor["total_bytes"] != max(1, end):
        raise ValueError("relay tensor layout does not match buffer length")


class RdmaWeightRelay:
    """Lease slices of one registered CPU arena for the engine's lifetime.

    Keeping the MR stable prevents cached peer descriptors from referring to an
    old rkey after allocator address reuse. Cancellation and native timeouts are
    not completion signals: only terminal transfer status permits slice reuse.
    The controller must settle downstream reads before releasing their sources.
    """

    def __init__(self, max_staging_bytes: int, device_name: str = "", staging_slot_bytes: int = 0):
        if type(max_staging_bytes) is not int or max_staging_bytes <= 0:
            raise ValueError("relay max_staging_bytes must be a positive integer")
        if not isinstance(device_name, str):
            raise TypeError("relay device_name must be a string")
        if (
            type(staging_slot_bytes) is not int
            or staging_slot_bytes < 0
            or staging_slot_bytes > max_staging_bytes
            or staging_slot_bytes % 16
        ):
            raise ValueError("invalid RDMA staging slot size")
        self.staging_slot_bytes = staging_slot_bytes
        try:
            from mooncake.engine import TransferEngine
        except ImportError as error:
            raise RuntimeError("mooncake_rdma requires Mooncake TransferEngine on every relay node") from error
        self.max_staging_bytes = max_staging_bytes
        self.node_id = ray.get_runtime_context().get_node_id()
        host = ray.util.get_node_ip_address().strip("[]")
        self.engine = TransferEngine()
        for method in ("batch_transfer_async_read", "transfer_check_status"):
            if not callable(getattr(self.engine, method, None)):
                raise RuntimeError(f"Mooncake RDMA relay requires {method}; install a compatible Mooncake wheel")
        ret = self.engine.initialize(host, "P2PHANDSHAKE", "rdma", device_name)
        if ret != 0:
            raise RuntimeError(f"Mooncake RDMA initialize failed ({ret}); check RDMA devices and permissions")
        self.session_id = f"{host}:{self.engine.get_rpc_port()}"
        self.device_name = device_name
        self._buckets = {}
        self._arena = None
        self._registered_bytes = 0
        self._active_bytes = 0
        self._pending_read = None
        self._lock = asyncio.Lock()
        self._closed = False

    def _require_open(self):
        if self._closed:
            raise RuntimeError("weight relay is closed")

    def _allocate(self, size):
        if self.staging_slot_bytes and size > self.staging_slot_bytes:
            raise ValueError(f"RDMA bucket {size} exceeds staging slot {self.staging_slot_bytes}")
        if size > self.max_staging_bytes - self._active_bytes:
            raise RuntimeError(
                f"relay staging capacity exceeded: {size} requested, "
                f"{self.max_staging_bytes - self._active_bytes} available"
            )
        if self._arena is None:
            arena = torch.empty(self.max_staging_bytes, dtype=torch.uint8, device="cpu")
            ret = self.engine.register_memory(arena.data_ptr(), arena.numel())
            if ret != 0:
                raise RuntimeError(f"Mooncake register_memory failed ({ret})")
            self._arena = arena
            self._registered_bytes = arena.numel()
            logger.info(
                "RDMA arena registered timestamp=%.6f session=%s pointer=%#x bytes=%d device=%s",
                time.time(),
                self.session_id,
                arena.data_ptr(),
                arena.numel(),
                self.device_name or "auto",
            )
        base = self._arena.data_ptr()
        offset = 0
        for buffer, descriptor in sorted(self._buckets.values(), key=lambda entry: entry[1]["pointer"]):
            start = descriptor["pointer"] - base
            if offset + size <= start:
                break
            offset = _aligned(start + (self.staging_slot_bytes or buffer.numel()))
        if offset + size > self._arena.numel():
            raise RuntimeError("relay staging capacity fragmented; increase max_staging_bytes or reduce pipeline_depth")
        return self._arena[offset : offset + size].zero_()

    def _register(self, buffer, metadata):
        pointer = buffer.data_ptr()
        descriptor = {
            **metadata,
            "format_version": 1,
            "bucket_id": uuid4().hex,
            "node_id": self.node_id,
            "session_id": self.session_id,
            "pointer": pointer,
            "total_bytes": buffer.numel(),
        }
        self._buckets[descriptor["bucket_id"]] = (buffer, descriptor)
        self._active_bytes += buffer.numel()
        logger.info(
            "RDMA bucket leased timestamp=%.6f session=%s bucket=%s pointer=%#x bytes=%d",
            time.time(),
            self.session_id,
            descriptor["bucket_id"],
            pointer,
            buffer.numel(),
        )
        return descriptor

    def _release(self, bucket_id):
        entry = self._buckets.get(bucket_id)
        if entry is None:
            return {"released": False, "bucket_id": bucket_id}
        buffer, descriptor = entry
        del self._buckets[bucket_id]
        self._active_bytes -= buffer.numel()
        logger.info(
            "RDMA bucket released timestamp=%.6f session=%s bucket=%s pointer=%#x bytes=%d",
            time.time(),
            self.session_id,
            bucket_id,
            descriptor["pointer"],
            buffer.numel(),
        )
        return {"released": True, "bucket_id": bucket_id}

    def _load(self, bucket):
        tensors, size = _layout(bucket)
        buffer = self._allocate(size)
        for (_, tensor), meta in zip(bucket, tensors, strict=True):
            if meta["nbytes"]:
                raw = tensor.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
                buffer[meta["offset"] : meta["offset"] + meta["nbytes"]].copy_(raw)
        return self._register(buffer, {"tensors": tensors, "checksum": _checksum(buffer)})

    def _pull(self, source):
        _validate_descriptor(source)
        buffer = self._allocate(source["total_bytes"])
        descriptor = self._register(buffer, {"tensors": source["tensors"], "checksum": source["checksum"]})
        try:
            self._read(source, descriptor)
            if _checksum(buffer) != source["checksum"]:
                raise ValueError("RDMA relay bucket checksum mismatch")
            # Default torch/Ray pickling includes a tensor's entire storage.
            # Publish compact independent tensors, never views of the arena.
            weights = tuple(
                (
                    meta["name"],
                    buffer[meta["offset"] : meta["offset"] + meta["nbytes"]]
                    .view(_DTYPES[meta["dtype"]])
                    .reshape(meta["shape"])
                    .clone(),
                )
                for meta in source["tensors"]
            )
            return {"descriptor": descriptor, "ref": ray.put(weights)}
        except BaseException:
            self._release(descriptor["bucket_id"])
            raise

    def _read(self, source, destination):
        started = time.monotonic()
        self._pending_read = {
            "source_bucket_id": source["bucket_id"],
            "source_session_id": source["session_id"],
            "source_pointer": source["pointer"],
            "destination_bucket_id": destination["bucket_id"],
            "destination_pointer": destination["pointer"],
            "bytes": source["total_bytes"],
            "started_at": time.time(),
            "native_status": None,
        }
        logger.info("RDMA read submitted session=%s read=%s", self.session_id, self._pending_read)
        try:
            # Exactly one request lets transfer_check_status safely free the batch
            # only after every slice is COMPLETED/FAILED. transfer_sync_read can
            # return -1 on its own timer while slices are still accessing memory.
            try:
                handle = self.engine.batch_transfer_async_read(
                    source["session_id"], [destination["pointer"]], [source["pointer"]], [source["total_bytes"]]
                )
            except BaseException as error:
                self._quarantine_read(f"native submission raised {type(error).__name__}: {error}")
            if not handle:
                # Mooncake can return 0 after a partial submit and a failed
                # freeBatchID(BatchBusy), discarding the only completion handle.
                self._quarantine_read("native submission returned no completion handle")
            next_warning = started + 30
            while True:
                try:
                    status = self.engine.transfer_check_status(handle)
                except BaseException as error:
                    self._quarantine_read(f"native completion query raised {type(error).__name__}: {error}")
                self._pending_read["native_status"] = status
                if status in (1, -1):
                    logger.info(
                        "RDMA read terminal session=%s source_bucket=%s destination_bucket=%s status=%s elapsed_s=%.3f",
                        self.session_id,
                        source["bucket_id"],
                        destination["bucket_id"],
                        status,
                        time.monotonic() - started,
                    )
                    if status == -1:
                        raise RuntimeError("Mooncake RDMA read failed (terminal status -1)")
                    return
                if status not in (0, -2):
                    self._quarantine_read(f"unknown native completion status: {status}")
                # 0 is pending, -2 is timeout without a completion guarantee.
                # Keep polling and retain both leases even if the RPC is cancelled.
                if time.monotonic() >= next_warning:
                    logger.warning(
                        "RDMA read still pending; retaining source/destination leases "
                        "session=%s elapsed_s=%.3f read=%s",
                        self.session_id,
                        time.monotonic() - started,
                        self._pending_read,
                    )
                    next_warning = time.monotonic() + 30
                time.sleep(0.01)
        finally:
            self._pending_read = None

    def _quarantine_read(self, error):
        # Returning from this RPC would allow the controller to release its
        # source too. No Python timeout/retry can establish native quiescence.
        self._pending_read.update(error=error, requires_process_restart=True)
        self._closed = True
        while True:
            logger.error(
                "RDMA completion is unknown; source/destination leases quarantined. "
                "Stop the affected relay processes before retrying. session=%s read=%s",
                self.session_id,
                self._pending_read,
            )
            time.sleep(30)

    async def _settle(self, operation, *args):
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        return task.result(), cancelled

    async def load_bucket(self, bucket):
        async with self._lock:
            self._require_open()
            descriptor, cancelled = await self._settle(self._load, bucket)
            if cancelled:
                await self._settle(self._release, descriptor["bucket_id"])
                raise asyncio.CancelledError
            return copy.deepcopy(descriptor)

    async def pull_bucket(self, descriptor):
        async with self._lock:
            self._require_open()
            result, cancelled = await self._settle(self._pull, copy.deepcopy(descriptor))
            if cancelled:
                await self._settle(self._release, result["descriptor"]["bucket_id"])
                raise asyncio.CancelledError
            result["descriptor"] = copy.deepcopy(result["descriptor"])
            return result

    async def release_bucket(self, bucket_id):
        async with self._lock:
            result, cancelled = await self._settle(self._release, bucket_id)
            if cancelled:
                raise asyncio.CancelledError
            return result

    async def get_status(self):
        # Diagnostics must remain available while a native read owns the lock.
        return {
            "node_id": self.node_id,
            "session_id": self.session_id,
            "registered_bytes": self._registered_bytes,
            "registered_buckets": len(self._buckets),
            "active_bytes": self._active_bytes,
            "pending_read": dict(self._pending_read) if self._pending_read else None,
            "max_staging_bytes": self.max_staging_bytes,
            "closed": self._closed,
        }

    def _unregister_arena(self):
        if self._arena is None:
            return
        pointer = self._arena.data_ptr()
        ret = self.engine.unregister_memory(pointer)
        if ret != 0:
            # Retain storage and its budget if the engine still owns the mapping.
            raise RuntimeError(f"Mooncake unregister_memory failed ({ret})")
        logger.info(
            "RDMA arena unregistered timestamp=%.6f session=%s pointer=%#x bytes=%d",
            time.time(),
            self.session_id,
            pointer,
            self._registered_bytes,
        )
        self._arena = None
        self._registered_bytes = 0

    async def close(self):
        async with self._lock:
            self._closed = True
            cancelled = False
            for bucket_id in list(self._buckets):
                _, was_cancelled = await self._settle(self._release, bucket_id)
                cancelled |= was_cancelled
            _, was_cancelled = await self._settle(self._unregister_arena)
            cancelled |= was_cancelled
            if cancelled:
                raise asyncio.CancelledError


RdmaWeightRelayActor = ray.remote(num_cpus=0, max_restarts=0)(RdmaWeightRelay)
