# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Real Ray actors and refs; a test-only socket engine substitutes for the RDMA NIC."""

import asyncio
import ctypes
import socket
import socketserver
import struct
import sys
import threading
from types import SimpleNamespace

import pytest
import ray

from tests.checkpoint_engine.test_rdma_relay_cluster import (
    RelayConsumer,
    SnapshotSource,
    assert_no_active_leases,
)
from verl.checkpoint_engine import relay_distribution
from verl.checkpoint_engine.rdma_relay import RdmaWeightRelay


def receive_exact(connection, count):
    chunks = []
    while count:
        part = connection.recv(count)
        if not part:
            raise RuntimeError("test transfer stream closed prematurely")
        chunks.append(part)
        count -= len(part)
    return b"".join(chunks)


class SocketTransferEngine:
    """Exercise real registered buffers across processes without a hardware claim."""

    def __init__(self):
        self.regions = {}
        self.fail_reads = False

    def initialize(self, hostname, metadata, protocol, device_name):
        assert metadata == "P2PHANDSHAKE" and protocol == "rdma"
        engine = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                pointer, size = struct.unpack("!QQ", receive_exact(self.request, 16))
                assert any(
                    base <= pointer and pointer + size <= base + length for base, length in engine.regions.items()
                )
                self.request.sendall(ctypes.string_at(pointer, size))

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return 0

    def get_rpc_port(self):
        return self.server.server_address[1]

    def register_memory(self, pointer, size):
        self.regions[pointer] = size
        return 0

    def unregister_memory(self, pointer):
        del self.regions[pointer]
        return 0

    def transfer_sync_read(self, session, pointer, source, size):
        if self.fail_reads:
            return -7
        assert any(base <= pointer and pointer + size <= base + length for base, length in self.regions.items())
        with socket.create_connection(("127.0.0.1", int(session.rsplit(":", 1)[1])), timeout=10) as connection:
            connection.sendall(struct.pack("!QQ", source, size))
            payload = receive_exact(connection, size)
        ctypes.memmove(pointer, payload, size)
        return 0

    def batch_transfer_async_read(self, session, pointers, sources, sizes):
        self.pending_read = (session, pointers[0], sources[0], sizes[0])
        return 1

    def transfer_check_status(self, handle):
        assert handle == 1
        return 1 if self.transfer_sync_read(*self.pending_read) == 0 else -1

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@ray.remote(num_cpus=0, max_restarts=0)
class SocketRelay(RdmaWeightRelay):
    def __init__(self, node, max_staging_bytes, device_name, fail_reads=False, staging_slot_bytes=0):
        sys.modules["mooncake.engine"] = SimpleNamespace(TransferEngine=SocketTransferEngine)
        super().__init__(
            max_staging_bytes=max_staging_bytes, device_name=device_name, staging_slot_bytes=staging_slot_bytes
        )
        self.node_id = node
        self.engine.fail_reads = fail_reads
        self.engine_closed = False

    async def close(self):
        await super().close()
        if not self.engine_closed:
            await asyncio.to_thread(self.engine.close)
            self.engine_closed = True


@pytest.fixture(scope="module")
def ray_runtime():
    ray.init(address="local", num_cpus=2, include_dashboard=False, object_store_memory=100_000_000)
    yield
    ray.shutdown()


@pytest.mark.parametrize("failed_node", [None, "middle"])
def test_relay_chain_preserves_local_snapshot_refs_and_survives_a_failed_middle_node(
    ray_runtime, monkeypatch, failed_node
):
    async def run():
        actors, relays = [], []
        distribution = None
        failed_nodes = {failed_node} if failed_node else set()

        def create_relay(node, config):
            actor = SocketRelay.remote(
                node,
                config["max_staging_bytes"],
                config["device_name"],
                fail_reads=node in failed_nodes,
                staging_slot_bytes=config["max_staging_bytes"] // config["pipeline_depth"] // 16 * 16,
            )
            relays.append(actor)
            return actor

        monkeypatch.setattr(relay_distribution, "create_relay", create_relay)
        try:
            source = SnapshotSource.remote()
            consumers = [RelayConsumer.remote(node) for node in ("middle", "tail")]
            actors.extend([source, *consumers])
            replicas = [SimpleNamespace(replica_rank=rank, workers=[worker]) for rank, worker in enumerate(consumers)]
            distribution = relay_distribution.RelayDistribution(
                replicas, "source", {"device_name": "", "max_staging_bytes": 1 << 20, "pipeline_depth": 2}
            )
            first = await source.export.remote(1)
            assert len(first.buckets) > 1
            assert first.chunks
            distribution.pin(first)
            # A newer manifest does not replace the version held by an installer.
            second = await source.export.remote(2)
            distribution.publish(second)
            for rank, (node, worker) in enumerate(zip(("middle", "tail"), consumers, strict=True)):
                if node == failed_node:
                    with pytest.raises(RuntimeError, match="failed"):
                        await asyncio.wait_for(distribution.snapshots_for(rank, 1), 30)
                    continue
                manifests = await asyncio.wait_for(distribution.snapshots_for(rank, 1), 30)
                assert manifests[node].chunks == first.chunks
                result = await worker.verify.remote(manifests[node], 1)
                assert result["tensor_count"] == 6
            await assert_no_active_leases(list(distribution.relays.values()))
            assert set(distribution.get_status()["node_errors"]) == ({failed_node} if failed_node else set())
            if failed_node:
                failed_nodes.clear()
                await asyncio.wait_for(distribution.retry_failed(0, 1), 30)
                recovered = await asyncio.wait_for(distribution.snapshots_for(0, 1), 30)
                assert (await consumers[0].verify.remote(recovered["middle"], 1))["version"] == 1
                assert not distribution.get_status()["node_errors"]
            distribution.unpin(first)
            distribution.pin(second)
            newest = await asyncio.wait_for(distribution.snapshots_for(1, 2), 30)
            assert (await consumers[1].verify.remote(newest["tail"], 2))["version"] == 2
            distribution.unpin(second)
            current_relays = list(distribution.relays.values())
            await assert_no_active_leases(current_relays)
            retired_relays = [relay for relay in relays if all(relay is not current for current in current_relays)]
            for relay in retired_relays:
                with pytest.raises(ray.exceptions.ActorDiedError):
                    await relay.get_status.remote()
            await distribution.close()
            assert distribution.closed and not distribution.relays
            for relay in relays:
                with pytest.raises(ray.exceptions.ActorDiedError):
                    await relay.get_status.remote()
            distribution = None
        finally:
            if distribution is not None:
                await distribution.close()
            for actor in [*relays, *actors]:
                ray.kill(actor, no_restart=True)

    asyncio.run(asyncio.wait_for(run(), 120))
