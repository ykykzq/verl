"""Payload-free TP layout manifests and bounded GPU snapshot version slots."""

import hashlib
import json
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPUChunk:
    tp_rank: int
    name: str
    owner: int
    offset: int
    numel: int
    dtype: str

    @property
    def key(self):
        return f"{self.tp_rank}:{self.name}:{self.offset}"


@dataclass(frozen=True)
class GPULayout:
    tensors: tuple
    chunks: tuple[GPUChunk, ...]
    owner_bytes: tuple[int, ...]
    layout_hash: str


@dataclass(frozen=True)
class GPUWeightSnapshot:
    version: int
    slot: int
    layout_hash: str


def tensor_dtype(name):
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported GPU snapshot dtype {name!r}")
    return dtype


def plan_layout(layouts, replicas, tp_size, chunk_bytes):
    if any(type(n) is not int or n <= 0 for n in (replicas, tp_size, chunk_bytes)):
        raise ValueError("replicas, TP size and chunk bytes must be positive integers")
    if len(layouts) != replicas * tp_size:
        raise ValueError("one runtime layout is required per rollout GPU")
    canonical = []
    for layout in layouts:
        entries = []
        names = set()
        for entry in layout:
            name, shape = entry["name"], tuple(entry["shape"])
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("runtime tensor names must be unique and nonempty")
            if any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError(f"invalid runtime shape for {name}")
            dtype = tensor_dtype(entry["dtype"])
            names.add(name)
            entries.append((name, shape, str(dtype)))
        if not entries:
            raise ValueError("runtime layout is empty")
        canonical.append(tuple(sorted(entries)))
    for rank, layout in enumerate(canonical):
        if layout != canonical[rank % tp_size]:
            raise ValueError(f"TP layout differs between replicas at rank {rank}")
    owner_bytes = [0] * len(layouts)
    chunks = []
    for tp_rank, layout in enumerate(canonical[:tp_size]):
        owners = list(range(tp_rank, len(layouts), tp_size))
        for name, shape, dtype in layout:
            itemsize = torch.empty((), dtype=tensor_dtype(dtype)).element_size()
            count = chunk_bytes // itemsize
            if count == 0:
                raise ValueError("chunk bytes cannot hold a tensor element")
            total = math.prod(shape)
            for offset in range(0, total, count):
                numel = min(count, total - offset)
                owner = min(owners, key=lambda i: (owner_bytes[i], i))
                owner_bytes[owner] += numel * itemsize
                chunks.append(GPUChunk(tp_rank, name, owner, offset, numel, dtype))
    payload = json.dumps([canonical, [(c.key, c.owner, c.numel) for c in chunks]], separators=(",", ":"))
    return GPULayout(
        tuple(canonical[:tp_size]), tuple(chunks), tuple(owner_bytes), hashlib.sha256(payload.encode()).hexdigest()
    )


class SnapshotSlots:
    """Latest and pinned versions survive supersession; staging is never evicted."""

    def __init__(self, count):
        if type(count) is not int or count < 2:
            raise ValueError("GPU snapshot publication needs at least two version slots")
        self.count = count
        self.versions = {}
        self.latest = None

    def reserve(self, version):
        if type(version) is not int or version < 0 or version in self.versions:
            raise ValueError("snapshot version must be new and nonnegative")
        if self.latest is not None and version <= self.latest:
            raise ValueError("snapshot versions must increase")
        used = {item["slot"] for item in self.versions.values()}
        free = next((i for i in range(self.count) if i not in used), None)
        if free is None:
            candidates = [v for v, s in self.versions.items() if v != self.latest and s["ready"] and not s["readers"]]
            if not candidates:
                raise BufferError("all GPU version slots are staging, latest, or pinned")
            free = self.versions.pop(min(candidates))["slot"]
        self.versions[version] = {"slot": free, "ready": False, "readers": 0}
        return free

    def commit(self, version):
        if version not in self.versions or self.versions[version]["ready"]:
            raise ValueError("snapshot is not staging")
        if self.latest is not None and version <= self.latest:
            raise ValueError("cannot commit a superseded snapshot")
        self.versions[version]["ready"] = True
        self.latest = version

    def abort(self, version):
        if version in self.versions and not self.versions[version]["ready"]:
            del self.versions[version]

    def pin(self, version):
        if version not in self.versions or not self.versions[version]["ready"]:
            raise ValueError("snapshot is not complete or has been evicted")
        self.versions[version]["readers"] += 1

    def unpin(self, version):
        if version not in self.versions or self.versions[version]["readers"] <= 0:
            raise ValueError("snapshot has no reader to release")
        self.versions[version]["readers"] -= 1
