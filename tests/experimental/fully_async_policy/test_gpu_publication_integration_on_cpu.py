import asyncio
from types import SimpleNamespace

import pytest

from tests.experimental.fully_async_policy.test_weight_publication_on_cpu import (
    RemoteMethod,
    Trainer,
    config,
    trainer,
    trainer_module,
)
from verl.checkpoint_engine.gpu_snapshot import GPUWeightSnapshot


@pytest.mark.asyncio
async def test_gpu_export_settles_all_training_ranks_before_seal_and_publishes_metadata():
    events = []
    release = asyncio.Event()
    snapshot = GPUWeightSnapshot(3, 0, "tp-layout")

    async def record(event, result=None):
        events.append(event)
        return result

    async def slow_rank():
        await release.wait()
        events.append("rank1-settled")

    async def publish(value):
        assert value is snapshot
        events.append("publish")

    instance = trainer(lambda **kwargs: pytest.fail("CPU export called"), publish)
    instance.config.actor_rollout_ref.rollout.weight_publication.transport = "nccl_gpu"
    instance.actor_rollout_wg.export_gpu_snapshot = lambda **kwargs: [
        record("rank0-export", {"source_export_s": 1}),
        slow_rank(),
    ]
    instance.weight_publisher.begin_gpu_snapshot = RemoteMethod(lambda version: record("reserve"))
    instance.weight_publisher.seal_gpu_snapshot = RemoteMethod(lambda version: record("seal", snapshot))
    task = asyncio.create_task(instance._export_and_publish_weights())
    await asyncio.sleep(0.02)
    assert "seal" not in events
    release.set()
    await task
    assert events == ["reserve", "rank0-export", "rank1-settled", "seal", "publish"]
    assert instance._last_published_version == 3


@pytest.mark.asyncio
async def test_gpu_setup_initializes_source_and_storage_concurrently(monkeypatch):
    from verl.checkpoint_engine import weight_publication

    source_ready, storage_ready = asyncio.Event(), asyncio.Event()
    info = {"world_size": 13, "unique_id": (1, 2), "timeout_s": 2}

    async def source(**kwargs):
        assert kwargs == info
        source_ready.set()
        await storage_ready.wait()

    async def storage():
        storage_ready.set()
        await source_ready.wait()

    handle = SimpleNamespace(
        configure_gpu=RemoteMethod(lambda publisher: asyncio.sleep(0, result=info)),
        initialize_gpu=RemoteMethod(storage),
    )
    calls = []
    monkeypatch.setattr(
        weight_publication,
        "WeightPublisherActor",
        SimpleNamespace(remote=lambda **kwargs: calls.append(kwargs) or handle),
    )
    instance = Trainer.__new__(Trainer)
    instance.config = config()
    instance.config.actor_rollout_ref.rollout.weight_publication.transport = "nccl_gpu"
    instance.rollouter = SimpleNamespace(
        get_replicas=RemoteMethod(lambda: asyncio.sleep(0, result=["replica"])),
        get_load_balancer=RemoteMethod(lambda: asyncio.sleep(0, result="router")),
    )
    instance.actor_rollout_wg = SimpleNamespace(initialize_gpu_snapshot=lambda **kwargs: [source(**kwargs)])
    await asyncio.wait_for(instance._setup_checkpoint_manager(), 2)
    assert calls[0]["gpu_snapshot_config"].slots == 2
    assert "relay_config" not in calls[0]


@pytest.mark.parametrize(
    "field,value",
    [("data_parallel_size", 2), ("pipeline_model_parallel_size", 2), ("quantization", "fp8"), ("dtype", "auto")],
)
def test_gpu_rejects_unimplemented_topologies_and_layouts(field, value):
    cfg = config()
    cfg.actor_rollout_ref.rollout.weight_publication.transport = "nccl_gpu"
    cfg.actor_rollout_ref.rollout[field] = value
    with pytest.raises(ValueError, match="nccl_gpu"):
        trainer_module.validate_weight_publication_config(cfg)


@pytest.mark.asyncio
async def test_gpu_export_failure_drains_collective_iterator(monkeypatch):
    import torch

    from verl.workers.engine_workers import ActorRolloutRefWorker

    consumed = []

    def weights():
        for i in range(4):
            consumed.append(i)
            yield str(i), torch.ones(2)

    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.role = "actor"
    worker.config = SimpleNamespace(rollout=SimpleNamespace(dtype="bfloat16"))
    worker.actor = SimpleNamespace(engine=SimpleNamespace(get_per_tensor_param=lambda: (weights(), None)))
    errors = []

    async def failed(error):
        errors.append(error)

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    publisher = SimpleNamespace(fail_gpu_snapshot=RemoteMethod(failed))
    with pytest.raises(ValueError, match="CUDA source"):
        await ActorRolloutRefWorker.export_gpu_snapshot.__wrapped__(worker, 1, publisher)
    assert consumed == [0, 1, 2, 3]
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_gpu_shutdown_quiesces_before_closing_source_and_storage_together(monkeypatch):
    events = []
    drained = asyncio.Event()
    source_closing = asyncio.Event()
    storage_closing = asyncio.Event()
    instance = trainer(None, None)
    instance.config.actor_rollout_ref.rollout.weight_publication.transport = "nccl_gpu"
    handle = instance.weight_publisher

    async def quiesce(drain):
        assert drain
        events.append("quiesce")
        await drained.wait()
        events.append("quiesced")

    async def source_close():
        assert "quiesced" in events
        source_closing.set()
        await storage_closing.wait()
        events.append("source_closed")

    async def storage_close():
        assert "quiesced" in events
        storage_closing.set()
        await source_closing.wait()
        events.append("storage_closed")

    handle.quiesce_gpu = RemoteMethod(quiesce)
    handle.close_gpu = RemoteMethod(storage_close)
    instance.actor_rollout_wg.close_gpu_snapshot = lambda: [source_close()]
    monkeypatch.setattr(trainer_module.ray, "kill", lambda actor: events.append("publisher_killed"))
    task = asyncio.create_task(instance.shutdown_weight_publisher())
    await asyncio.sleep(0.02)
    assert events == ["quiesce"]
    drained.set()
    await asyncio.wait_for(task, 1)
    assert set(events[2:4]) == {"source_closed", "storage_closed"}
    assert events[-1] == "publisher_killed"
    assert instance.weight_publisher is None


@pytest.mark.asyncio
async def test_cancelled_gpu_shutdown_still_settles_both_communicator_owners(monkeypatch):
    released = asyncio.Event()
    started = asyncio.Event()
    events = []
    instance = trainer(None, None)
    instance.config.actor_rollout_ref.rollout.weight_publication.transport = "nccl_gpu"

    async def quiesce(drain):
        started.set()
        await released.wait()

    async def record(event):
        events.append(event)

    instance.weight_publisher.quiesce_gpu = RemoteMethod(quiesce)
    instance.weight_publisher.close_gpu = RemoteMethod(lambda: record("storage"))
    instance.actor_rollout_wg.close_gpu_snapshot = lambda: [record("source")]
    monkeypatch.setattr(trainer_module.ray, "kill", lambda actor: events.append("killed"))
    task = asyncio.create_task(instance.shutdown_weight_publisher())
    await asyncio.wait_for(started.wait(), 1)
    for _ in range(2):
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
    released.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert set(events[:2]) == {"source", "storage"}
    assert events[-1] == "killed"


@pytest.mark.asyncio
async def test_failed_gpu_quiesce_retains_source_and_storage(monkeypatch):
    instance = trainer(None, None)
    instance.config.actor_rollout_ref.rollout.weight_publication.transport = "nccl_gpu"
    handle = instance.weight_publisher

    async def quiesce(drain):
        raise RuntimeError("GPU transport failed; restart workers")

    handle.quiesce_gpu = RemoteMethod(quiesce)
    handle.close_gpu = RemoteMethod(lambda: pytest.fail("uncertain storage must remain owned"))
    instance.actor_rollout_wg.close_gpu_snapshot = lambda: pytest.fail("uncertain source must remain owned")
    kills = []
    monkeypatch.setattr(trainer_module.ray, "kill", lambda actor: kills.append(actor))
    with pytest.raises(RuntimeError, match="restart workers"):
        await instance.shutdown_weight_publisher()
    assert kills == [handle]


@pytest.mark.asyncio
async def test_public_publisher_close_quiesces_pool_after_scheduler_drain():
    from verl.checkpoint_engine.weight_publication import WeightPublisher

    events = []

    async def record(event, **kwargs):
        events.append(event)

    publisher = WeightPublisher.__new__(WeightPublisher)
    publisher.scheduler = SimpleNamespace(close=lambda **kwargs: record("scheduler", **kwargs))
    publisher.relay = None
    publisher.gpu_pool = SimpleNamespace(quiesce=lambda: record("quiesce"), close=lambda: record("storage_close"))
    await publisher.close()
    assert events == ["scheduler", "quiesce", "storage_close"]


@pytest.mark.asyncio
async def test_source_close_allows_fresh_initialization_but_rejects_live_reinitialization(monkeypatch):
    import torch

    from verl.checkpoint_engine import gpu_snapshot_transport
    from verl.workers.engine_workers import ActorRolloutRefWorker

    created = []

    class Transport:
        def __init__(self, timeout_s):
            self.closed = False
            created.append(self)

        async def initialize(self, *args):
            pass

        async def close(self):
            self.closed = True

    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(gpu_snapshot_transport, "DirectedNCCL", Transport)
    initialize = ActorRolloutRefWorker.initialize_gpu_snapshot.__wrapped__
    await initialize(worker, 2, (1,), 1)
    with pytest.raises(RuntimeError, match="already|restart"):
        await initialize(worker, 2, (2,), 1)
    assert worker._gpu_snapshot_sender is created[0]
    await ActorRolloutRefWorker.close_gpu_snapshot.__wrapped__(worker)
    assert created[0].closed
    await initialize(worker, 2, (3,), 1)
    assert len(created) == 2
    assert worker._gpu_snapshot_sender is created[1]


@pytest.mark.asyncio
async def test_source_failed_close_keeps_native_owner(monkeypatch):
    import torch

    from verl.workers.engine_workers import ActorRolloutRefWorker

    async def close():
        raise RuntimeError("uncertain transfer; worker restart required")

    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    sender = SimpleNamespace(close=close)
    worker._gpu_snapshot_sender = sender
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    with pytest.raises(RuntimeError, match="restart"):
        await ActorRolloutRefWorker.close_gpu_snapshot.__wrapped__(worker)
    assert worker._gpu_snapshot_sender is sender


@pytest.mark.asyncio
async def test_source_releases_acknowledged_cast_before_next_fsdp_tensor(monkeypatch):
    import weakref

    import torch

    from verl.workers.engine_workers import ActorRolloutRefWorker

    payloads = []

    class Payload:
        device = SimpleNamespace(type="cuda")
        shape = (2,)
        dtype = "torch.bfloat16"

        def is_floating_point(self):
            return False

        def contiguous(self):
            return self

    class SourceTensor:
        def detach(self):
            value = Payload()
            payloads.append(weakref.ref(value))
            return value

    def weights():
        yield "first", SourceTensor()
        assert payloads[0]() is None, "previous cast overlaps the next FSDP all-gather"
        yield "second", SourceTensor()

    async def send(payload, destination):
        assert destination == 1

    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.role = "actor"
    worker.config = SimpleNamespace(rollout=SimpleNamespace(dtype="bfloat16"))
    worker.actor = SimpleNamespace(engine=SimpleNamespace(get_per_tensor_param=lambda: (weights(), None)))
    worker._gpu_snapshot_sender = SimpleNamespace(transfer=send)
    publisher = SimpleNamespace(
        prepare_gpu_input=RemoteMethod(lambda *args: asyncio.sleep(0, result=[1])),
        finish_gpu_input=RemoteMethod(lambda: asyncio.sleep(0)),
        fail_gpu_snapshot=RemoteMethod(lambda message: asyncio.sleep(0)),
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    await ActorRolloutRefWorker.export_gpu_snapshot.__wrapped__(worker, 1, publisher)
    assert all(reference() is None for reference in payloads)
