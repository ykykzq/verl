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

import asyncio
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import ray
import torch

from verl.checkpoint_engine.weight_snapshot import WeightSnapshot, export_weight_snapshot, iter_snapshot_weights


class _LocalObjectRef(ray.ObjectRef):
    """An awaitable ObjectRef that retains payloads without a Ray daemon."""

    values = {}
    fetched = []

    def __await__(self):
        async def fetch():
            self.fetched.append(self.binary())
            return self.values[self.binary()]

        return fetch().__await__()


@pytest.fixture
def object_store(monkeypatch):
    _LocalObjectRef.values = {}
    _LocalObjectRef.fetched = []

    def put(value):
        key = (len(_LocalObjectRef.values) + 1).to_bytes(28, "big")
        _LocalObjectRef.values[key] = value
        return _LocalObjectRef(key)

    monkeypatch.setattr(ray, "put", put)
    yield _LocalObjectRef
    _LocalObjectRef.values = {}
    _LocalObjectRef.fetched = []


def _consume(snapshot):
    async def consume():
        return [item async for item in iter_snapshot_weights(snapshot)]

    return asyncio.run(consume())


def test_export_detaches_cpu_storage_from_live_parameters(object_store):
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    snapshot = export_weight_snapshot(iter([("w", parameter)]), version=3, bucket_size=8, is_master=True)
    with torch.no_grad():
        parameter.fill_(99)

    name, saved = _consume(snapshot)[0]
    assert name == "w"
    assert saved.tolist() == [1.0, 2.0]
    assert saved.device.type == "cpu"
    assert saved.requires_grad is False
    assert snapshot.version == 3
    assert snapshot.tensor_count == 1
    assert snapshot.total_bytes == 8
    assert isinstance(snapshot.buckets, tuple)
    with pytest.raises(FrozenInstanceError):
        snapshot.version = 4


def test_rdma_snapshot_splits_oversized_tensor_and_restores_shape(object_store):
    weights = [
        ("embedding", torch.arange(105, dtype=torch.float32).reshape(15, 7)),
        ("bias", torch.arange(3, dtype=torch.float16)),
        ("scalar", torch.tensor(2.0)),
    ]
    snapshot = export_weight_snapshot(weights, 3, 64, True, split_tensors=True)
    assert snapshot.max_bucket_bytes <= 64
    assert len(snapshot.buckets) > 6
    for ref in snapshot.buckets:
        from verl.checkpoint_engine.rdma_relay import _layout

        assert _layout(object_store.values[ref.binary()])[1] <= 64
    recovered = _consume(snapshot)
    for (name, value), (actual_name, actual) in zip(weights, recovered, strict=True):
        assert name == actual_name
        assert torch.equal(value, actual)


def test_chunked_snapshot_rejects_truncated_tensor(object_store):
    from dataclasses import replace

    snapshot = export_weight_snapshot([("large", torch.arange(40))], 3, 64, True, split_tensors=True)
    broken = replace(snapshot, buckets=snapshot.buckets[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        _consume(broken)


def test_consumer_mutation_does_not_change_snapshot_retry(object_store):
    snapshot = export_weight_snapshot([("w", torch.tensor([5, 6]))], 1, 64, True)
    _consume(snapshot)[0][1].fill_(0)
    assert _consume(snapshot)[0][1].tolist() == [5, 6]


def test_buckets_round_trip_and_only_oversized_tensor_exceeds_target(object_store):
    weights = [("a", torch.tensor([1.0, 2.0])), ("b", torch.tensor([3.0, 4.0]))]
    weights += [("large", torch.arange(5, dtype=torch.float32)), ("c", torch.tensor([9.0]))]
    snapshot = export_weight_snapshot(weights, 2, 12, True)
    sizes = [
        sum(t.numel() * t.element_size() for _, t in object_store.values[ref.binary()]) for ref in snapshot.buckets
    ]
    assert sizes == [8, 8, 20, 4]
    assert snapshot.total_bytes == 40
    assert snapshot.tensor_count == 4
    consumed = _consume(snapshot)
    assert [name for name, _ in consumed] == ["a", "b", "large", "c"]
    assert [tensor.tolist() for _, tensor in consumed] == [[1.0, 2.0], [3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0], [9.0]]
    assert object_store.fetched == [ref.binary() for ref in snapshot.buckets]


def test_consumer_fetches_buckets_lazily(object_store):
    snapshot = export_weight_snapshot([("a", torch.ones(2)), ("b", torch.ones(2))], 0, 8, True)

    async def consume_first():
        iterator = iter_snapshot_weights(snapshot)
        assert object_store.fetched == []
        name, _ = await anext(iterator)
        assert name == "a"
        assert object_store.fetched == [snapshot.buckets[0].binary()]
        await iterator.aclose()

    asyncio.run(consume_first())


def test_nonmaster_consumes_all_collective_entries_without_storing(monkeypatch):
    consumed = []

    def weights():
        for i in range(4):
            consumed.append(i)
            yield str(i), object()  # Non-source ranks need not materialize tensors.

    def unexpected_put(_value):
        pytest.fail("Non-source rank must not publish buckets")

    monkeypatch.setattr(ray, "put", unexpected_put)
    assert export_weight_snapshot(weights(), 4, 16, False) is None
    assert consumed == [0, 1, 2, 3]


@pytest.mark.parametrize("failure", ["put", "materialize"])
def test_local_export_failure_drains_remaining_collective_entries(monkeypatch, failure):
    consumed = []

    class UncopyableTensor(torch.Tensor):
        def to(self, *args, **kwargs):
            raise RuntimeError("CPU copy failed")

    def weights():
        for i in range(4):
            consumed.append(i)
            tensor = torch.ones(2)
            if failure == "materialize" and i == 0:
                tensor = tensor.as_subclass(UncopyableTensor)
            yield str(i), tensor

    def failed_put(_value):
        raise RuntimeError("object store full")

    monkeypatch.setattr(ray, "put", failed_put)
    with pytest.raises(RuntimeError, match="object store full" if failure == "put" else "CPU copy failed"):
        export_weight_snapshot(weights(), 1, 8, True)
    assert consumed == [0, 1, 2, 3]


@pytest.mark.parametrize("version,bucket_size", [(-1, 8), (True, 8), (1, 0)])
def test_invalid_export_metadata_still_drains_collective_entries(version, bucket_size):
    consumed = []

    def weights():
        for i in range(3):
            consumed.append(i)
            yield str(i), torch.ones(1)

    with pytest.raises(ValueError):
        export_weight_snapshot(weights(), version, bucket_size, True)
    assert consumed == [0, 1, 2]


def test_empty_export_and_duplicate_names_are_rejected(object_store):
    with pytest.raises(ValueError, match="empty"):
        export_weight_snapshot([], 0, 8, True)
    with pytest.raises(ValueError, match="Duplicate"):
        export_weight_snapshot([("w", torch.ones(1)), ("w", torch.ones(1))], 0, 8, True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"version": -1},
        {"version": True},
        {"buckets": []},
        {"buckets": ()},
        {"tensor_count": 0},
        {"total_bytes": -1},
        {"buckets": (object(),)},
    ],
)
def test_manifest_rejects_invalid_metadata(object_store, kwargs):
    fields = dict(version=0, buckets=(ray.put((("w", torch.ones(1)),)),), tensor_count=1, total_bytes=4)
    fields.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        WeightSnapshot(**fields)


@pytest.mark.parametrize(
    "payload,count,byte_count",
    [
        ((), 1, 4),
        ((("w", "not a tensor"),), 1, 4),
        ((("w", torch.ones(1)),), 2, 4),
        ((("w", torch.ones(1)),), 1, 8),
        ((("w", torch.ones(1)), ("w", torch.ones(1))), 2, 8),
    ],
)
def test_consumer_rejects_malformed_bucket_or_totals(object_store, payload, count, byte_count):
    snapshot = WeightSnapshot(0, (ray.put(payload),), count, byte_count)
    with pytest.raises((TypeError, ValueError)):
        _consume(snapshot)


def _worker(weights, peft_config=None):
    from verl.workers.engine_workers import ActorRolloutRefWorker

    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.role = "actor"
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1))
    )
    worker.actor = SimpleNamespace(engine=SimpleNamespace(get_per_tensor_param=lambda: (iter(weights), peft_config)))
    return worker


@pytest.mark.parametrize("rank", [0, 1])
def test_worker_exports_collectively_with_only_rank_zero_manifest(object_store, monkeypatch, rank):
    from verl.workers.engine_workers import ActorRolloutRefWorker

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    worker = _worker([("w", torch.tensor([7.0]))])
    snapshot = asyncio.run(ActorRolloutRefWorker.export_weight_snapshot.__wrapped__(worker, global_steps=9))
    if rank == 0:
        assert snapshot.version == 9
        assert _consume(snapshot)[0][1].tolist() == [7.0]
    else:
        assert snapshot is None
        assert object_store.values == {}


def test_worker_rejects_nonmerged_lora_export(object_store, monkeypatch):
    from verl.workers.engine_workers import ActorRolloutRefWorker

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    worker = _worker([("adapter", torch.ones(1))], peft_config={"r": 4})
    with pytest.raises(ValueError, match="LoRA"):
        asyncio.run(ActorRolloutRefWorker.export_weight_snapshot.__wrapped__(worker, global_steps=1))
    assert object_store.values == {}
