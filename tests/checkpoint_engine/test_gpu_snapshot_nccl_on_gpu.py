# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Opt-in CUDA/NCCL acceptance with synthetic runtime tensors, without RTP."""

import contextlib
import json
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VERL_RUN_GPU_SNAPSHOT_TEST") != "1",
    reason="requires explicit opt-in, CUDA Ray actors, CuPy and NCCL; not a CPU or real RTP TP test",
)


def runtime_layout():
    return [
        {"name": "float_bits", "shape": [5, 7], "dtype": "torch.float32"},
        {"name": "indices", "shape": [17], "dtype": "torch.int64"},
        {"name": "half", "shape": [33], "dtype": "torch.float16"},
    ]


def expected_tensor(shape, dtype_name, version, tp_rank, salt=0):
    import math

    import torch

    from verl.checkpoint_engine.gpu_snapshot import tensor_dtype

    dtype = tensor_dtype(dtype_name)
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    # Reinterpret deterministic bytes, including arbitrary floating-point bit
    # patterns. A numeric cast or allclose comparison cannot certify this path.
    raw = torch.arange(nbytes, dtype=torch.int64, device="cuda")
    raw = ((raw + version * 17 + tp_rank * 31 + salt) % 251).to(torch.uint8)
    return raw.view(dtype).reshape(shape)


class LocalAdapter:
    """Only satisfy worker construction; these tests never invoke RTP APIs."""

    use_shm = False
    server_handle = None

    def _ensure_server_handle(self):
        return True


class GPUAcceptanceActor:
    def __init__(self, rank, tp_size, timeout_s):
        import ray
        import torch

        from verl.checkpoint_engine.gpu_snapshot_transport import GPUSnapshotWorker

        if not torch.cuda.is_available():
            raise RuntimeError("GPU snapshot acceptance actor has no CUDA device")
        torch.cuda.set_device(0)
        self.rank = rank
        self.tp_rank = rank % tp_size
        self.worker = GPUSnapshotWorker(
            LocalAdapter(),
            SimpleNamespace(
                operation_timeout_s=timeout_s,
                max_bytes_per_gpu=1 << 20,
                max_tensor_bytes=1 << 20,
                max_conversion_bytes=1 << 20,
            ),
        )
        self.identity = {
            "rank": rank,
            "node_id": ray.get_runtime_context().get_node_id(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_ids": ray.get_gpu_ids(),
            "device_name": torch.cuda.get_device_name(0),
        }

    def describe(self):
        return {"identity": self.identity, "layout": runtime_layout()}

    def unique_id(self):
        return self.worker.transport.unique_id()

    async def initialize(self, world_size, unique_id):
        await self.worker.initialize(world_size, unique_id, self.rank)

    async def send_raw(self, dtype_name, peer):
        import torch

        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            tensor = expected_tensor((8193,), dtype_name, 7, 0)
            await self.worker.transport.transfer(tensor, peer)
        return {"dtype": str(tensor.dtype), "bytes": tensor.nbytes, "device": tensor.device.type}

    async def receive_raw(self, dtype_name, peer):
        import hashlib

        import torch

        from verl.checkpoint_engine.gpu_snapshot import tensor_dtype

        tensor = torch.empty((8193,), dtype=tensor_dtype(dtype_name), device="cuda")
        await self.worker.transport.transfer(tensor, peer, receive=True)
        expected = expected_tensor((8193,), dtype_name, 7, 0)
        assert torch.equal(tensor.view(torch.uint8), expected.view(torch.uint8)), dtype_name
        return {
            "dtype": str(tensor.dtype),
            "verified_bytes": tensor.nbytes,
            "sha256": hashlib.sha256(tensor.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
        }

    async def prepare(self, slot, version):
        import torch

        await self.worker.reserve(slot)
        self.worker.outputs = {
            entry["name"]: expected_tensor(entry["shape"], entry["dtype"], version, self.tp_rank, index)
            for index, entry in enumerate(runtime_layout())
        }
        torch.cuda.synchronize()

    async def store_chunk(self, slot, chunk, source):
        await self.worker.store_chunk(slot, chunk, source)

    async def send_output(self, chunk, destination):
        await self.worker.send_output(chunk, destination)

    async def allocate(self, key, shape, dtype):
        await self.worker.allocate(key, shape, dtype)

    async def receive_chunk(self, key, slot, chunk, source):
        await self.worker.receive_chunk(key, slot, chunk, source)

    async def send_chunk(self, slot, chunk, destination):
        await self.worker.send_chunk(slot, chunk, destination)

    def verify(self, version):
        import torch

        total = 0
        for index, entry in enumerate(runtime_layout()):
            key = f"verify:{version}:{entry['name']}"
            actual = self.worker.buffers[key]
            expected = expected_tensor(entry["shape"], entry["dtype"], version, self.tp_rank, index)
            assert actual.device.type == "cuda"
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)), key
            total += actual.nbytes
            del self.worker.buffers[key]
        return {**self.identity, "version": version, "tp_rank": self.tp_rank, "verified_bytes": total}


@contextlib.contextmanager
def gpu_actors(replicas, tp_size):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    connected_here = not ray.is_initialized()
    actors = []
    timeout = float(os.environ.get("VERL_GPU_SNAPSHOT_TEST_TIMEOUT_S", "120"))
    try:
        if connected_here:
            ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
        nodes = sorted(
            [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) >= 1],
            key=lambda node: node["NodeID"],
        )
        if not nodes:
            pytest.fail("Explicit GPU acceptance requested, but Ray has no live CUDA GPU nodes")
        remaining = [int(node["Resources"]["GPU"]) for node in nodes]
        world_size = replicas * tp_size
        if sum(remaining) < world_size:
            pytest.skip(f"synthetic TP-column case requires {world_size} Ray GPUs; available total={sum(remaining)}")
        require_multinode = os.environ.get("VERL_GPU_SNAPSHOT_REQUIRE_MULTINODE") == "1"
        if require_multinode and len(nodes) < 2:
            pytest.fail("Cross-node GPU acceptance requested, but Ray has fewer than two GPU nodes")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("NCCL_", "TORCH_NCCL_"))
            or key in ("PYTORCH_ALLOC_CONF", "LD_LIBRARY_PATH", "PYTHONPATH")
        }
        actor_type = ray.remote(
            num_gpus=1,
            num_cpus=0,
            max_restarts=0,
            runtime_env={"env_vars": environment},
        )(GPUAcceptanceActor)
        for rank in range(world_size):
            # Prefer one replica per node so owners in a TP column communicate
            # across hosts. Fall back only when a node has too few GPUs.
            preferred = (rank // tp_size) % len(nodes)
            choices = [preferred] + [index for index in range(len(nodes)) if index != preferred]
            index = next(index for index in choices if remaining[index])
            remaining[index] -= 1
            actors.append(
                actor_type.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[index]["NodeID"], soft=False)
                ).remote(rank, tp_size, timeout)
            )
        descriptions = ray.get([actor.describe.remote() for actor in actors], timeout=timeout)
        identities = [item["identity"] for item in descriptions]
        if require_multinode:
            assert len({identity["node_id"] for identity in identities}) >= 2
        unique_id = ray.get(actors[0].unique_id.remote(), timeout=timeout)
        ray.get([actor.initialize.remote(world_size, unique_id) for actor in actors], timeout=timeout)
        yield ray, actors, descriptions, timeout
    finally:
        # Process death owns cleanup after NCCL timeout. Never clear a timed-out
        # worker's tensors or attempt to reuse its communicator.
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if connected_here and ray.is_initialized():
            ray.shutdown()


def test_directed_nccl_integer_and_float_bytes():
    with gpu_actors(replicas=2, tp_size=1) as (ray, actors, descriptions, timeout):
        for dtype in ("torch.int64", "torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"):
            for source, destination in ((0, 1), (1, 0)):
                sent, received = ray.get(
                    [
                        actors[source].send_raw.remote(dtype, destination),
                        actors[destination].receive_raw.remote(dtype, source),
                    ],
                    timeout=timeout,
                )
                assert sent["device"] == "cuda" and sent["bytes"] == received["verified_bytes"]
                print(
                    json.dumps(
                        {
                            "case": "directed_bytes",
                            "source": source,
                            "destination": destination,
                            "actors": [item["identity"] for item in descriptions],
                            **received,
                        }
                    ),
                    flush=True,
                )


@pytest.mark.parametrize("tp_size", [1, 2])
def test_gpu_shard_columns_and_pinned_versions(tp_size):
    from verl.checkpoint_engine.gpu_snapshot import GPUWeightSnapshot, SnapshotSlots, plan_layout

    with gpu_actors(replicas=2, tp_size=tp_size) as (ray, actors, descriptions, timeout):
        plan = plan_layout([item["layout"] for item in descriptions], replicas=2, tp_size=tp_size, chunk_bytes=32)
        assert all(chunk.owner % tp_size == chunk.tp_rank for chunk in plan.chunks)
        assert all(plan.owner_bytes), "fixture must store chunks on both replicas"
        slots = SnapshotSlots(2)

        def publish(version):
            slot = slots.reserve(version)
            ray.get([actor.prepare.remote(slot, version) for actor in actors], timeout=timeout)
            for chunk in plan.chunks:
                source = chunk.tp_rank
                if chunk.owner == source:
                    ray.get(actors[source].store_chunk.remote(slot, chunk, None), timeout=timeout)
                else:
                    ray.get(
                        [
                            actors[source].send_output.remote(chunk, chunk.owner),
                            actors[chunk.owner].store_chunk.remote(slot, chunk, source),
                        ],
                        timeout=timeout,
                    )
            slots.commit(version)
            return GPUWeightSnapshot(version, slot, plan.layout_hash)

        def fetch(snapshot):
            assert snapshot.layout_hash == plan.layout_hash
            for destination, actor in enumerate(actors):
                for name, shape, dtype in plan.tensors[destination % tp_size]:
                    key = f"verify:{snapshot.version}:{name}"
                    ray.get(actor.allocate.remote(key, shape, dtype), timeout=timeout)
                    for chunk in plan.chunks:
                        if chunk.tp_rank != destination % tp_size or chunk.name != name:
                            continue
                        if chunk.owner == destination:
                            ray.get(actor.receive_chunk.remote(key, snapshot.slot, chunk, None), timeout=timeout)
                        else:
                            ray.get(
                                [
                                    actors[chunk.owner].send_chunk.remote(snapshot.slot, chunk, destination),
                                    actor.receive_chunk.remote(key, snapshot.slot, chunk, chunk.owner),
                                ],
                                timeout=timeout,
                            )
                result = ray.get(actor.verify.remote(snapshot.version), timeout=timeout)
                assert result["verified_bytes"] == 342
                print(
                    json.dumps(
                        {"case": "synthetic_tp_layout", "tp_size": tp_size, "layout_hash": plan.layout_hash, **result}
                    ),
                    flush=True,
                )

        first = publish(7)
        slots.pin(first.version)
        second = publish(8)
        with pytest.raises(BufferError, match="pinned"):
            slots.reserve(9)
        fetch(first)
        fetch(second)
        slots.unpin(first.version)
        third = publish(9)
        assert third.slot == first.slot
        with pytest.raises(ValueError, match="evicted"):
            slots.pin(first.version)
        fetch(third)
        fetch(second)
