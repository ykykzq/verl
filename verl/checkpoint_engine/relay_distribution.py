# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Bounded CPU relay chains with independent node readiness and version leases."""

import asyncio

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.checkpoint_engine.weight_snapshot import WeightSnapshot


def create_relay(node_id, config):
    from verl.checkpoint_engine.rdma_relay import RdmaWeightRelayActor

    return RdmaWeightRelayActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
    ).remote(max_staging_bytes=config["max_staging_bytes"], device_name=config["device_name"])


def destroy_relay(handle):
    ray.kill(handle, no_restart=True)


async def _settle(task):
    """A cancelled Ray await cannot prove that a remote RDMA read stopped."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class _Distribution:
    def __init__(self, snapshot, source, nodes, relays, depth):
        self.snapshot, self.source, self.nodes, self.relays = snapshot, source, nodes, relays
        self.depth = depth
        self.stopping = False
        self.errors = {}
        self.source_error = False
        self.cleanup_errors = []
        self.buckets = {node: [None] * len(snapshot.buckets) for node in nodes}
        self.ready = {node: asyncio.get_running_loop().create_future() for node in nodes}
        self.task = asyncio.create_task(self.run())

    def failed(self, node, error):
        self.errors[node] = str(error)
        if node in self.ready and not self.ready[node].done():
            self.ready[node].set_result(str(error))

    def received(self, node, index, ref):
        if node in self.errors:
            return
        if not isinstance(ref, ray.ObjectRef):
            raise TypeError("relay must return a nested local bucket ObjectRef")
        self.buckets[node][index] = ref
        if all(ref is not None for ref in self.buckets[node]) and not self.ready[node].done():
            self.ready[node].set_result(
                WeightSnapshot(
                    self.snapshot.version,
                    tuple(self.buckets[node]),
                    self.snapshot.tensor_count,
                    self.snapshot.total_bytes,
                )
            )

    async def bucket(self, index):
        registered = []
        try:
            descriptor = await self.relays[self.source].load_bucket.remote(self.snapshot.buckets[index])
            registered.append((self.source, descriptor["bucket_id"]))
            if descriptor["node_id"] != self.source:
                raise RuntimeError("master relay was placed on the wrong node")
            predecessors = [descriptor]
            for node in self.nodes:
                if self.stopping:
                    break
                if node in self.errors:
                    continue
                if node == self.source:
                    self.received(node, index, self.snapshot.buckets[index])
                    continue
                try:
                    # A predecessor may die after advertising its completed bucket.
                    # Every earlier source stays registered until this chain settles.
                    for predecessor in reversed(predecessors):
                        try:
                            result = await self.relays[node].pull_bucket.remote(predecessor)
                            break
                        except Exception:
                            if predecessor is predecessors[0]:
                                raise
                    following = result["descriptor"]
                    registered.append((node, following["bucket_id"]))
                    if following["node_id"] != node:
                        raise RuntimeError("destination relay was placed on the wrong node")
                    self.received(node, index, result["ref"])
                    predecessors.append(following)
                except Exception as error:
                    self.failed(node, error)
                    # The last successful descriptor stays live for downstream readers.
        except Exception as error:
            self.source_error = True
            for node in self.nodes:
                self.failed(node, error)
        finally:
            results = await asyncio.gather(
                *(self.relays[node].release_bucket.remote(bucket_id) for node, bucket_id in registered),
                return_exceptions=True,
            )
            for (node, _), result in zip(registered, results, strict=True):
                if isinstance(result, ray.exceptions.ActorDiedError):
                    self.failed(node, result)
                    self.source_error |= node == self.source
                elif isinstance(result, BaseException):
                    self.cleanup_errors.append(str(result))

    async def run(self):
        indices = iter(range(len(self.snapshot.buckets)))

        async def lane():
            for index in indices:
                if self.stopping:
                    break
                await self.bucket(index)

        lanes = asyncio.gather(*(lane() for _ in range(min(self.depth, len(self.snapshot.buckets)))))
        try:
            await asyncio.shield(lanes)
        except asyncio.CancelledError:
            self.stopping = True
            await _settle(lanes)
        finally:
            for node, ready in self.ready.items():
                if not ready.done():
                    ready.set_result("relay distribution superseded or closed")

    async def close(self):
        self.stopping = True
        await _settle(self.task)
        self.buckets.clear()
        self.ready.clear()


class RelayDistribution:
    """Keep one local version, one active lease, and the latest source manifest.

    New publications coalesce while a replica installs. Replacing a distribution
    waits for its already-issued reads; it never frees a remote source early.
    """

    def __init__(self, replicas, source_node_id, config):
        if not isinstance(source_node_id, str) or not source_node_id:
            raise ValueError("RDMA publication requires the export source node ID")
        self.replicas, self.source = replicas, source_node_id
        self.config = dict(config)
        self.relays = {}
        self.replica_nodes = {}
        self.nodes = []
        self.latest = self.pinned = self.current = None
        self.error = None
        self.closed = False
        self._wake = asyncio.Event()
        self._changed = asyncio.Condition()
        self._task = None
        self._restart = None

    def _start(self):
        if self.closed:
            raise RuntimeError("weight relay is closed")
        if self.error:
            raise RuntimeError(f"weight relay failed: {self.error}")
        self._wake.set()
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="weight-relay-distribution")

    def publish(self, snapshot):
        snapshot.__post_init__()
        if self.latest is None or snapshot.version > self.latest.version:
            self.latest = snapshot
        self._start()

    def pin(self, snapshot):
        if self.pinned is not None:
            raise RuntimeError("another replica already owns the relay installation lease")
        self.publish(snapshot)
        self.pinned = snapshot

    def unpin(self, snapshot):
        if self.pinned is not snapshot:
            raise RuntimeError("relay installation lease does not match snapshot")
        self.pinned = None
        self._wake.set()

    async def _initialize(self):
        for replica in self.replicas:
            nodes = await asyncio.gather(*(w.get_weight_relay_node_id.remote() for w in replica.workers))
            if not nodes or any(not isinstance(node, str) or not node for node in nodes):
                raise ValueError("checkpoint workers did not report valid relay node IDs")
            self.replica_nodes[replica.replica_rank] = set(nodes)
        self.nodes = sorted(set().union(*self.replica_nodes.values()))
        for node in dict.fromkeys([self.source, *self.nodes]):
            self.relays[node] = create_relay(node, self.config)

    async def _notify(self):
        async with self._changed:
            self._changed.notify_all()

    async def retry_failed(self, replica_rank, version):
        current = self.current
        if current is None or current.snapshot.version != version:
            return
        if not any(node in current.errors for node in self.replica_nodes[replica_rank]):
            return
        self._restart = current
        self._wake.set()
        async with self._changed:
            while self.current is current:
                if self.error or self.closed:
                    raise RuntimeError(self.error or "weight relay is closed")
                await self._changed.wait()

    async def _run(self):
        try:
            await self._initialize()
            while not self.closed:
                self._wake.clear()
                desired = self.pinned or self.latest
                if desired is not None and (
                    self.current is None
                    or self.current.snapshot.version != desired.version
                    or self._restart is self.current
                ):
                    if self.current is not None:
                        await self.current.close()
                        if self.current.cleanup_errors:
                            raise RuntimeError(f"relay cleanup failed: {self.current.cleanup_errors}")
                        failed = set(self.current.errors)
                        if self.current.source_error:
                            failed.add(self.source)
                        for node in sorted(failed):
                            try:
                                await self.relays[node].close.remote()
                            except Exception:
                                pass  # The dead actor is replaced after every outstanding read settled.
                            destroy_relay(self.relays[node])
                            self.relays[node] = create_relay(node, self.config)
                        self.current = None
                    self._restart = None
                    if self.closed:
                        break
                    desired = self.pinned or self.latest
                    self.current = _Distribution(
                        desired, self.source, self.nodes, self.relays, self.config["pipeline_depth"]
                    )
                    await self._notify()
                await self._wake.wait()
        except Exception as error:
            self.error = str(error)
        finally:
            await self._notify()

    async def snapshots_for(self, replica_rank, version):
        async with self._changed:
            while self.current is None or self.current.stopping or self.current.snapshot.version != version:
                if self.error or self.closed:
                    raise RuntimeError(self.error or "weight relay is closed")
                await self._changed.wait()
            current = self.current
            nodes = sorted(self.replica_nodes[replica_rank])
            futures = [current.ready[node] for node in nodes]
        snapshots = await asyncio.gather(*(asyncio.shield(future) for future in futures))
        for node, snapshot in zip(nodes, snapshots, strict=True):
            if node in current.errors:
                raise RuntimeError(f"weight relay on node {node} failed: {current.errors[node]}")
            if not isinstance(snapshot, WeightSnapshot):
                raise RuntimeError(f"weight relay on node {node} failed: {snapshot}")
        return dict(zip(nodes, snapshots, strict=True))

    def get_status(self):
        current = self.current
        return {
            "transport": "mooncake_rdma",
            "source_node_id": self.source,
            "active_version": current.snapshot.version if current else None,
            "pinned_version": self.pinned.version if self.pinned else None,
            "latest_version": self.latest.version if self.latest else None,
            "ready_nodes": [
                n
                for n, f in current.ready.items()
                if n not in current.errors and f.done() and isinstance(f.result(), WeightSnapshot)
            ]
            if current
            else [],
            "node_errors": dict(current.errors) if current else {},
            "fatal_error": self.error or (str(current.cleanup_errors) if current and current.cleanup_errors else None),
            "closed": self.closed,
        }

    async def close(self):
        self.closed = True
        self._wake.set()
        if self._task is not None:
            await _settle(self._task)
        if self.current is not None:
            await self.current.close()
        closing = asyncio.gather(*(relay.close.remote() for relay in self.relays.values()), return_exceptions=True)
        results = await _settle(closing)
        await self._notify()
        failures = [
            str(result)
            for result in results
            if isinstance(result, BaseException) and not isinstance(result, ray.exceptions.ActorDiedError)
        ]
        if failures:
            raise RuntimeError(f"weight relay shutdown failed: {failures}")
        for relay in self.relays.values():
            destroy_relay(relay)
        self.current = self.latest = self.pinned = self._restart = None
        self.relays.clear()
