# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from verl.checkpoint_engine import base


@pytest.fixture
def nccl(monkeypatch):
    # Exercise the real engine lifecycle while replacing only GPU allocation and
    # collective transport, which are unavailable in this CPU test environment.
    cupy = ModuleType("cupy")
    cupy.ndarray = torch.Tensor
    cupy.uint8 = torch.uint8
    cupy.zeros = torch.zeros
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(base.CheckpointEngineRegistry, "_registry", dict(base.CheckpointEngineRegistry._registry))
    spec = importlib.util.spec_from_file_location(
        "_temporary_nccl_test", Path(base.__file__).with_name("nccl_checkpoint_engine.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    groups, events = {}, []

    def create(world_size, rank, backend, name):
        assert name not in groups, "previous replica communicator was not destroyed"
        groups[name] = (world_size, rank)
        events.append("create")

    def destroy(name):
        del groups[name]
        events.append("destroy")

    module.collective = SimpleNamespace(
        is_group_initialized=lambda name: name in groups,
        init_collective_group=create,
        destroy_collective_group=destroy,
        barrier=lambda name: None,
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    engine = module.NCCLCheckpointEngine.__new__(module.NCCLCheckpointEngine)
    engine.bucket_size = 8
    engine.is_master = True
    engine.multi_sender = False
    engine.group_name = "weights"
    engine.rebuild_group = False
    engine.ip, engine.listen_port = "127.0.0.1", 1234
    monkeypatch.setattr(engine, "get_node_id", lambda: "node")
    return SimpleNamespace(engine=engine, groups=groups, events=events)


def test_switching_replicas_rebuilds_communicator_even_with_default_engine_config(nccl):
    engine = nccl.engine
    for _ in range(2):
        metadata = engine.prepare_temporary()
        engine.init_process_group(0, 2, metadata.master, num_senders=1)
        engine.finalize()
    assert nccl.events == ["create", "destroy", "create", "destroy"]
    assert nccl.groups == {}


def test_temporary_prepare_discards_an_existing_cached_group(nccl):
    engine = nccl.engine
    metadata = engine.prepare()
    engine.init_process_group(0, 2, metadata.master, num_senders=1)
    engine.finalize()
    assert nccl.groups == {"weights": (2, 0)}
    metadata = engine.prepare_temporary()
    engine.init_process_group(0, 3, metadata.master, num_senders=1)
    engine.finalize()
    assert nccl.events == ["create", "destroy", "create", "destroy"]


def test_finalize_is_safe_after_partial_prepare_and_repeated_cleanup(nccl, monkeypatch):
    engine = nccl.engine
    engine.rebuild_group = True

    def allocation_failure(*args, **kwargs):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(sys.modules["cupy"], "zeros", allocation_failure)
    with pytest.raises(RuntimeError, match="allocation failed"):
        engine.prepare()
    engine.finalize()
    engine.finalize()
    assert engine.send_buf is None and engine.recv_buf is None
    assert nccl.groups == {}


def test_temporary_finalize_closes_consumer_metadata_socket(nccl):
    engine = nccl.engine
    events = []
    engine.is_master = False
    engine.rebuild_group = True
    engine.socket = SimpleNamespace(
        close=lambda linger: events.append(("close", linger)),
        context=SimpleNamespace(term=lambda: events.append("term")),
    )
    engine.finalize()
    assert events == [("close", 0), "term"]
    assert engine.socket is None
