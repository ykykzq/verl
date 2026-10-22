import pytest

from verl.checkpoint_engine.gpu_snapshot import SnapshotSlots, plan_layout


def test_gpu_publication_config_is_opt_in_and_bounded():
    from verl.workers.config.rollout import WeightPublicationConfig

    config = WeightPublicationConfig(enabled=True, transport="nccl_gpu", gpu_snapshot={"slots": 3})
    assert config.gpu_snapshot.slots == 3
    for kwargs in ({"slots": 1}, {"max_bytes_per_gpu": 0}, {"operation_timeout_s": float("inf")}):
        with pytest.raises(ValueError):
            WeightPublicationConfig(enabled=True, transport="nccl_gpu", gpu_snapshot=kwargs)
    with pytest.raises(ValueError):
        WeightPublicationConfig(transport="nccl_gpu")


def layouts():
    return [[{"name": "layer", "shape": [10], "dtype": "torch.float32"}]] * 4


def test_tp_column_placement_and_exact_coverage():
    plan = plan_layout(layouts(), replicas=2, tp_size=2, chunk_bytes=16)
    for tp in range(2):
        chunks = [c for c in plan.chunks if c.tp_rank == tp]
        assert [(c.offset, c.numel) for c in chunks] == [(0, 4), (4, 4), (8, 2)]
        assert all(c.owner % 2 == tp for c in chunks)
    assert sum(plan.owner_bytes) == 80
    assert plan.layout_hash == plan_layout(layouts(), 2, 2, 16).layout_hash


def test_layout_rejects_cross_replica_mismatch_and_bad_chunk():
    broken = layouts()
    broken[2] = [{"name": "layer", "shape": [11], "dtype": "torch.float32"}]
    with pytest.raises(ValueError, match="TP layout"):
        plan_layout(broken, 2, 2, 16)
    with pytest.raises(ValueError):
        plan_layout(layouts(), 2, 2, 0)


def test_slots_pin_latest_and_readers_and_abort_staging():
    slots = SnapshotSlots(2)
    first = slots.reserve(1)
    slots.commit(1)
    slots.pin(1)
    slots.reserve(2)
    slots.commit(2)
    with pytest.raises(BufferError):
        slots.reserve(3)
    slots.unpin(1)
    assert slots.reserve(3) == first
    slots.abort(3)
    assert slots.latest == 2
    with pytest.raises(ValueError):
        slots.reserve(2)
    with pytest.raises(ValueError):
        slots.pin(3)


def test_failed_commit_and_staging_cannot_be_evicted():
    slots = SnapshotSlots(2)
    slots.reserve(0)
    slots.reserve(1)
    with pytest.raises(BufferError):
        slots.reserve(2)
    with pytest.raises(ValueError):
        slots.commit(2)
    slots.commit(1)
    with pytest.raises(ValueError):
        slots.commit(0)
