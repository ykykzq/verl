# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

import asyncio
import copy
import ctypes
import sys
import threading
from types import SimpleNamespace

import pytest
import ray
import torch


class FakeTransferEngine:
    instances = []

    def __init__(self):
        self.regions = {}
        self.read_error = 0
        self.register_error = 0
        self.unregister_error = 0
        self.read_started = threading.Event()
        self.read_continue = None
        self.instances.append(self)

    def initialize(self, hostname, metadata, protocol, device_name):
        assert metadata == "P2PHANDSHAKE" and protocol == "rdma"
        return 0

    def get_rpc_port(self):
        return 10000 + self.instances.index(self)

    def register_memory(self, pointer, size):
        if not self.register_error:
            self.regions[pointer] = size
        return self.register_error

    def unregister_memory(self, pointer):
        if not self.unregister_error:
            self.regions.pop(pointer)
        return self.unregister_error

    def transfer_sync_read(self, session, pointer, source, size):
        self.read_started.set()
        if self.read_continue is not None:
            assert self.read_continue.wait(5)
        if self.read_error:
            return self.read_error
        assert self.regions[pointer] >= size
        assert any(engine.regions.get(source, 0) >= size for engine in self.instances)
        ctypes.memmove(pointer, source, size)
        return 0


@pytest.fixture
def runtime(monkeypatch):
    FakeTransferEngine.instances = []
    monkeypatch.setitem(sys.modules, "mooncake.engine", SimpleNamespace(TransferEngine=FakeTransferEngine))
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "node"))
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    values = {}

    def put(value):
        ref = ray.ObjectRef((len(values) + 1).to_bytes(28, "big"))
        values[ref] = tuple((name, tensor.clone()) for name, tensor in value)
        return ref

    monkeypatch.setattr(ray, "put", put)
    return values


def relay(budget=4096):
    from verl.checkpoint_engine.rdma_relay import RdmaWeightRelay

    return RdmaWeightRelay(budget)


@pytest.mark.asyncio
async def test_relay_round_trip_mixed_dtypes_shapes_and_source_immutability(runtime):
    source, target = relay(), relay()
    weights = (
        ("matrix", torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T),
        ("scalar", torch.tensor(8, dtype=torch.int64)),
        ("empty", torch.empty(0, 2)),
        ("flags", torch.tensor([True, False])),
    )
    expected = tuple((name, tensor.clone()) for name, tensor in weights)
    descriptor = await source.load_bucket(weights)
    weights[0][1].fill_(99)
    result = await target.pull_bucket(descriptor)
    assert isinstance(result["ref"], ray.ObjectRef)
    for (name, tensor), (expected_name, expected_tensor) in zip(runtime[result["ref"]], expected, strict=True):
        assert name == expected_name
        assert tensor.dtype == expected_tensor.dtype and tensor.shape == expected_tensor.shape
        assert torch.equal(tensor, expected_tensor)
    assert result["descriptor"]["bucket_id"] != descriptor["bucket_id"]
    assert result["descriptor"]["checksum"] == descriptor["checksum"]
    await source.release_bucket(descriptor["bucket_id"])
    await target.release_bucket(result["descriptor"]["bucket_id"])
    assert (await source.get_status())["registered_bytes"] == 0
    assert (await target.get_status())["registered_bytes"] == 0


@pytest.mark.asyncio
async def test_all_empty_tensors_still_have_valid_registered_storage(runtime):
    source, target = relay(), relay()
    descriptor = await source.load_bucket((("empty", torch.empty(2, 0)),))
    assert descriptor["total_bytes"] >= 1
    result = await target.pull_bucket(descriptor)
    assert runtime[result["ref"]][0][1].shape == (2, 0)
    await source.close()
    await target.close()


@pytest.mark.asyncio
async def test_capacity_is_bounded_and_released_only_after_deregistration(runtime):
    source = relay(16)
    descriptor = await source.load_bucket((("a", torch.ones(4)),))
    with pytest.raises(RuntimeError, match="capacity|budget"):
        await source.load_bucket((("b", torch.ones(1)),))
    engine = FakeTransferEngine.instances[0]
    engine.unregister_error = -1
    with pytest.raises(RuntimeError, match="unregister"):
        await source.release_bucket(descriptor["bucket_id"])
    assert (await source.get_status())["registered_bytes"] == 16
    engine.unregister_error = 0
    await source.release_bucket(descriptor["bucket_id"])
    assert (await source.get_status())["registered_bytes"] == 0


@pytest.mark.asyncio
async def test_failed_reads_and_checksum_mismatch_do_not_publish_or_leak(runtime):
    source, target = relay(), relay()
    descriptor = await source.load_bucket((("a", torch.ones(2)),))
    engine = FakeTransferEngine.instances[1]
    engine.read_error = -9
    with pytest.raises(RuntimeError, match="read"):
        await target.pull_bucket(descriptor)
    assert not engine.regions and not runtime
    engine.read_error = 0
    with pytest.raises(ValueError, match="checksum"):
        await target.pull_bucket({**descriptor, "checksum": "0" * 64})
    assert not engine.regions and not runtime
    await source.close()


@pytest.mark.asyncio
async def test_invalid_layout_is_rejected_before_allocating_or_reading(runtime):
    source, target = relay(), relay()
    descriptor = await source.load_bucket((("a", torch.ones(2)),))
    bad = copy.deepcopy(descriptor)
    bad["tensors"][0]["offset"] = descriptor["total_bytes"]
    with pytest.raises(ValueError, match="layout|offset|bounds"):
        await target.pull_bucket(bad)
    assert not FakeTransferEngine.instances[1].regions
    assert not FakeTransferEngine.instances[1].read_started.is_set()
    await source.close()


@pytest.mark.asyncio
async def test_cancelled_read_keeps_buffer_registered_until_real_transfer_finishes(runtime):
    source, target = relay(), relay()
    descriptor = await source.load_bucket((("a", torch.ones(2)),))
    engine = FakeTransferEngine.instances[1]
    engine.read_continue = threading.Event()
    task = asyncio.create_task(target.pull_bucket(descriptor))
    await asyncio.to_thread(engine.read_started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and engine.regions
    task.cancel()
    engine.read_continue.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not engine.regions
    await source.close()


@pytest.mark.asyncio
async def test_failed_registration_does_not_charge_capacity(runtime):
    source = relay()
    FakeTransferEngine.instances[0].register_error = -1
    with pytest.raises(RuntimeError, match="register"):
        await source.load_bucket((("a", torch.ones(2)),))
    assert (await source.get_status())["registered_bytes"] == 0


@pytest.mark.asyncio
async def test_close_releases_all_buffers_and_prevents_new_transfers(runtime):
    source = relay()
    await source.load_bucket((("a", torch.ones(2)),))
    await source.load_bucket((("b", torch.ones(2)),))
    await source.close()
    assert not FakeTransferEngine.instances[0].regions
    with pytest.raises(RuntimeError, match="clos"):
        await source.load_bucket((("c", torch.ones(2)),))


@pytest.mark.asyncio
async def test_invalid_named_tensors_never_register_memory(runtime):
    source = relay()
    for weights in [(), (("", torch.ones(1)),), (("a", 1),), (("a", torch.ones(1)), ("a", torch.ones(1)))]:
        with pytest.raises((TypeError, ValueError)):
            await source.load_bucket(weights)
    assert not FakeTransferEngine.instances[0].regions


@pytest.mark.asyncio
async def test_source_descriptor_is_not_an_alias_of_registered_metadata(runtime):
    source = relay()
    descriptor = await source.load_bucket((("a", torch.ones(2)),))
    bucket_id = descriptor["bucket_id"]
    descriptor["pointer"] = 1
    descriptor["tensors"][0]["name"] = "mutated"
    await source.release_bucket(bucket_id)
    assert not FakeTransferEngine.instances[0].regions


@pytest.mark.asyncio
async def test_put_failure_cleans_registered_receive_memory(runtime, monkeypatch):
    source, target = relay(), relay()
    descriptor = await source.load_bucket((("a", torch.ones(2)),))

    def failed_put(value):
        raise RuntimeError("object store unavailable")

    monkeypatch.setattr(ray, "put", failed_put)
    with pytest.raises(RuntimeError, match="object store unavailable"):
        await target.pull_bucket(descriptor)
    assert not FakeTransferEngine.instances[1].regions
    await source.close()


def test_missing_mooncake_is_explicit_and_never_uses_a_fallback(runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "mooncake.engine", None)
    with pytest.raises(RuntimeError, match="requires Mooncake"):
        relay()


def test_failed_rdma_initialization_is_explicit(runtime, monkeypatch):
    monkeypatch.setattr(FakeTransferEngine, "initialize", lambda *args: -3)
    with pytest.raises(RuntimeError, match="RDMA initialize failed"):
        relay()


@pytest.mark.asyncio
async def test_repeated_close_can_retry_failed_unregistration(runtime):
    source = relay()
    await source.load_bucket((("a", torch.ones(2)),))
    engine = FakeTransferEngine.instances[0]
    engine.unregister_error = -1
    with pytest.raises(RuntimeError, match="unregister"):
        await source.close()
    assert engine.regions
    with pytest.raises(RuntimeError, match="closed"):
        await source.load_bucket((("b", torch.ones(2)),))
    engine.unregister_error = 0
    await source.close()
    assert not engine.regions
