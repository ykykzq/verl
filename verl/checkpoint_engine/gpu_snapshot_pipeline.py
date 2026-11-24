"""Bounded receive/convert/distribute pipeline for immutable GPU snapshots.

The input broadcast reuses NCCLCheckpointEngine's multi-sender topology. Input,
distribution and per-TP fetch have independent communicators. Completion of a
CUDA event (not enqueue) gates every storage reuse and IPC release.
"""

import asyncio
import time

import torch

from verl.checkpoint_engine.gpu_snapshot_bucket import (
    BucketSnapshotPool,
    BucketSnapshotWorker,
    byte_layout,
)
from verl.checkpoint_engine.gpu_snapshot_pool import _VISION_PREFIX_RE, settle
from verl.checkpoint_engine.gpu_snapshot_transport import DirectedNCCL


def select_converters(source_nodes, worker_nodes, tp_size):
    if not source_nodes or not all(source_nodes) or len(worker_nodes) % tp_size:
        raise ValueError("invalid snapshot node topology")
    # Prefer the source's NVLink-local node, then another trainer node. Stable
    # tie-breaking is only used between workers with equal locality.
    return [
        min(
            range(tp, len(worker_nodes), tp_size),
            key=lambda i: (worker_nodes[i] != source_nodes[0], worker_nodes[i] not in source_nodes, i),
        )
        for tp in range(tp_size)
    ]


class PipelineSnapshotPool(BucketSnapshotPool):
    def __init__(self, replicas, config):
        super().__init__(replicas, config)
        self.input_credit = asyncio.Semaphore(2)
        self.output_credit = asyncio.Semaphore(2)
        self.jobs = set()
        self.convert_tail = None
        self.distribute_tail = None
        self.sequence = 0
        self.input_live = self.output_live = 0

    async def describe(self, source_nodes):
        from verl.checkpoint_engine.nccl_checkpoint_engine import (
            MasterMetadata,
            NCCLCheckpointEngine,
            WorkerMetadata,
        )

        await super().describe()
        worker_nodes = await settle([self.call(i, "node_id") for i in range(len(self.workers))])
        if getattr(self.config, "memory_planning", "legacy") == "role":
            from verl.checkpoint_engine.snapshot_memory import fixed_converter_workers

            self.converter_workers = fixed_converter_workers(
                self.config, [r.replica_rank for r in self.replicas], self.tp_size
            )
        else:
            self.converter_workers = select_converters(source_nodes, worker_nodes, self.tp_size)
        master = MasterMetadata(zmq_ip="", zmq_port=0, multi_sender=True)
        metadata = [WorkerMetadata(node_id=n, master=master if i == 0 else None) for i, n in enumerate(source_nodes)]
        metadata += [WorkerMetadata(node_id=worker_nodes[i]) for i in self.converter_workers]
        actor, rollout = NCCLCheckpointEngine.build_topology(len(source_nodes), self.tp_size, metadata)
        self.input_uid = await self.call(self.converter_workers[0], "unique_id")
        self.input_ranks = rollout["rank"]
        self.input_world = rollout["world_size"][0]
        self.topology = dict(
            source_nodes=source_nodes,
            worker_nodes=worker_nodes,
            converter_workers=self.converter_workers,
            actor_ranks=actor["rank"],
            num_senders=actor["num_senders"][0],
            input_world_size=self.input_world,
        )
        return dict(
            world_size=self.input_world,
            unique_id=self.input_uid,
            timeout_s=self.config.operation_timeout_s,
            actor_ranks=actor["rank"],
            pipeline=True,
        )

    async def configure_conversion_plans(self, source_layouts):
        from dataclasses import asdict

        from verl.checkpoint_engine.conversion_layout import merge_source_layouts, plan_source_buckets

        if self.initialized or self.staging is not None:
            raise RuntimeError("conversion plans must be validated before transport initialization")
        sources = merge_source_layouts(source_layouts)
        buckets = plan_source_buckets(sources, self.config.bucket_bytes, self.config.max_tensor_bytes)
        self.conversion_plans = []
        for tp in range(self.tp_size):
            transport = {
                "tp_rank": tp,
                "tp_size": self.tp_size,
                "source_buckets": buckets,
                "output_chunks": [asdict(c) for c in self.plan.chunks if c.tp_rank == tp],
                "owner_order": sorted({c.owner for c in self.plan.chunks if c.tp_rank == tp}),
                "input_topology": self.topology,
                "buffer_budgets": {
                    name: getattr(self.config, name)
                    for name in (
                        "max_bytes_per_gpu",
                        "max_tensor_bytes",
                        "max_conversion_bytes",
                        "bucket_bytes",
                        "pipeline_depth",
                        "slots",
                    )
                },
                "capture": "export_all_ranks_settled_before_next_optimizer_update",
            }
            descriptions = await settle(
                [
                    self.call(i, "prepare_conversion_plan", sources, transport)
                    for i in range(tp, len(self.workers), self.tp_size)
                ]
            )
            if len({d["plan_hash"] for d in descriptions}) != 1:
                raise ValueError("RTP replicas disagree on static conversion rules")
            self.conversion_plans.append(descriptions[0])
        self.source_layout = sources
        return [
            {"tp_rank": i, "plan_hash": d["plan_hash"], "rule_construction_s": d["rule_construction_s"]}
            for i, d in enumerate(self.conversion_plans)
        ]

    async def initialize(self):
        self.require_open()
        if self.initialized:
            raise RuntimeError("snapshot pool already initialized")
        try:
            await settle(
                [self.call(i, "initialize", len(self.workers), self.unique_id, i) for i in range(len(self.workers))]
            )
            await settle(
                [
                    self.call(i, "initialize_input", self.input_world, self.input_uid, rank)
                    for i, rank in zip(self.converter_workers, self.input_ranks, strict=True)
                ]
            )
            ids = await settle([self.call(tp, "unique_id") for tp in range(self.tp_size)])
            await settle(
                [
                    self.call(i, "initialize_fetch", len(self.replicas), ids[i % self.tp_size], i // self.tp_size)
                    for i in range(len(self.workers))
                ]
            )
        except BaseException as error:
            self.fatal_error = f"pipeline initialization failed: {error}"
            raise
        self.initialized = True

    async def begin(self, version):
        if self.jobs or self.input_live or self.output_live:
            raise RuntimeError("previous pipeline still owns staging buffers")
        result = await super().begin(version)
        self.sequence = 0
        self.convert_tail = self.distribute_tail = None
        if getattr(self, "conversion_plans", None):
            self.metrics["conversion_plan_hashes"] = [p["plan_hash"] for p in self.conversion_plans]
        self.metrics.update(topology=self.topology, pipeline_events=[], input_high_water=0, output_high_water=0)
        return result

    def track(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.jobs.add(task)

        def completed(done):
            self.jobs.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error:
                    self.fatal_error = self.fatal_error or f"pipeline stage failed: {error}"

        task.add_done_callback(completed)
        return task

    async def stage(self, name, number, calls):
        start = time.perf_counter()
        wall = time.time()
        try:
            return await settle(calls)
        finally:
            duration = time.perf_counter() - start
            self.metrics[name + "_wall_s"] = self.metrics.get(name + "_wall_s", 0) + duration
            self.metrics["pipeline_events"].append(dict(stage=name, bucket=number, start=wall, duration=duration))

    async def prepare_bucket(self, version, tensors):
        self.require_open()
        if self.staging is None or self.staging.version != version or self.ingest_task is not None:
            raise ValueError("bucket does not match idle input of staging version")
        entries, size = byte_layout(tensors)
        if not entries or size > self.config.max_tensor_bytes:
            raise ValueError("empty or oversized pipeline input")
        for name, *_ in entries:
            if name in self.exported_names:
                raise ValueError(f"duplicate exported source tensor {name}")
            if not any(name in required for required in self.required):
                if name not in self.ignored_source_names and not _VISION_PREFIX_RE.match(name):
                    raise ValueError(f"unexpected source tensor {name}")
        await self.input_credit.acquire()
        try:
            # Serialize admission with shutdown, but never hold this lock while
            # waiting for an input credit or for conversion/distribution.
            async with self.io_lock:
                self.require_open()
                number = self.sequence
                key = f"pipeline-input:{number % 2}"
                try:
                    await settle(
                        [self.call(i, "allocate", key, (size,), "torch.uint8") for i in self.converter_workers]
                    )
                except BaseException as error:
                    self.fatal_error = f"pipeline input allocation failed: {error}"
                    raise
                self.sequence += 1
                self.input_live += 1
                self.metrics["input_high_water"] = max(self.metrics["input_high_water"], self.input_live)
                self.exported_names.update(e[0] for e in entries)
                selected = [[e for e in entries if e[0] in required] for required in self.required]
                self.seen.update(e[0] for group in selected for e in group)
                receive = self.track(
                    self.stage(
                        "receive", number, [self.call(i, "receive_pipeline_input", key) for i in self.converter_workers]
                    )
                )
                previous = self.convert_tail
                self.convert_tail = self.track(self.convert_stage(number, key, selected, receive, previous))
                self.ingest_task = receive
                self.metrics["source_buckets"] = self.sequence
                return {"bucket": number, "broadcast": True}
        except BaseException:
            self.input_credit.release()
            raise

    async def finish_input(self):
        if self.ingest_task is None:
            raise RuntimeError("no pipeline input pending")
        await settle([self.ingest_task])
        self.ingest_task = None
        self.healthy()

    async def convert_stage(self, number, key, entries, received, previous, is_last=False):
        credit = False
        try:
            if received is not None:
                await settle([received])
            if previous is not None:
                await settle([previous])
            self.healthy()
            await self.output_credit.acquire()
            credit = True
            self.output_live += 1
            self.healthy()
            self.metrics["output_high_water"] = max(self.metrics["output_high_water"], self.output_live)
            output_key = f"pipeline-output:{number % 2}"
            outputs = await self.stage(
                "convert",
                number,
                [
                    self.call(i, "convert_pipeline_bucket", key, selected, output_key, is_last)
                    for i, selected in zip(self.converter_workers, entries, strict=True)
                ],
            )
            # Conversion completion includes its CUDA event: input can now be reused.
            previous_distribution = self.distribute_tail
            self.distribute_tail = self.track(self.distribute_stage(number, output_key, outputs, previous_distribution))
            credit = False  # distribution owns the output credit from this point
        except BaseException as error:
            self.fatal_error = self.fatal_error or f"pipeline conversion failed: {error}"
            raise
        finally:
            if key is not None:
                self.input_live -= 1
                self.input_credit.release()
            if credit:
                self.output_live -= 1
                self.output_credit.release()

    async def distribute_stage(self, number, key, outputs, previous):
        try:
            if previous is not None:
                await settle([previous])
            self.healthy()
            await self.stage(
                "distribute", number, [self.distribute_column(tp, key, group) for tp, group in enumerate(outputs)]
            )
        except BaseException as error:
            self.fatal_error = self.fatal_error or f"pipeline distribution failed: {error}"
            raise
        finally:
            self.output_live -= 1
            self.output_credit.release()

    async def distribute_column(self, tp, key, outputs):
        converter = self.converter_workers[tp]
        expected = {name: (shape, dtype) for name, shape, dtype in self.plan.tensors[tp]}
        names = set()
        for item in outputs:
            name = item["name"]
            if (tp, name) in self.completed or name in names:
                raise ValueError(f"duplicate converted tensor {(tp, name)}")
            if expected.get(name) != (tuple(item["shape"]), item["dtype"]):
                raise ValueError(f"converted layout mismatch {(tp, name)}")
            names.add(name)
        chunks = [c for c in self.plan.chunks if c.tp_rank == tp and c.name in names]
        for owner in sorted({c.owner for c in chunks}):
            group = [c for c in chunks if c.owner == owner]
            if owner == converter:
                await self.call(owner, "store_pipeline_chunks", self.staging.slot, group, None, key)
            else:
                await settle(
                    [
                        self.call(converter, "send_pipeline_outputs", group, owner, key),
                        self.call(owner, "store_pipeline_chunks", self.staging.slot, group, converter, key),
                    ]
                )
        self.completed.update((tp, name) for name in names)
        await self.call(converter, "release_pipeline_outputs", key)

    async def drain(self):
        # A conversion can enqueue its distribution while we are waiting.
        while self.jobs:
            await settle(list(self.jobs))
        self.healthy()

    async def _convert_columns_and_store(self, tp_ranks, name=None, is_last=False):
        if not is_last:
            raise RuntimeError("pipeline requires bucket inputs")
        await self.drain()
        await self.convert_stage(self.sequence, None, [[] for _ in range(self.tp_size)], None, None, True)
        await self.drain()

    async def seal(self, version):
        await self.drain()
        return await super().seal(version)

    async def _drain_ingestion(self):
        # Base quiesce has fenced admission. Wait for any admitted allocation
        # to hand its buffers to stage tasks before closing the converters.
        async with self.io_lock:
            await super()._drain_ingestion()
            await self.drain()


class PipelineSnapshotWorker(BucketSnapshotWorker):
    def __init__(self, adapter, config):
        super().__init__(adapter, config)
        self.input_transport = DirectedNCCL(config.operation_timeout_s)
        self.output_batches = {}

    async def node_id(self):
        import ray

        return ray.get_runtime_context().get_node_id()

    async def initialize_input(self, size, uid, rank):
        await self.input_transport.initialize(size, uid, rank)

    def _release_buffer(self, key):
        if not self.input_transport.failed:
            super()._release_buffer(key)

    async def receive_pipeline_input(self, key):
        await self.input_transport.broadcast(self.buffers[key])

    async def convert_pipeline_bucket(self, key, entries, output_key, is_last=False):
        from verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer import rebuild_ipc_tensor, serialize_ipc_tensor

        if output_key in self.output_batches or len(self.output_batches) >= 2:
            raise RuntimeError("pipeline output buffer is still in use")
        payload = serialize_ipc_tensor(self.buffers[key]) if entries else None
        try:
            result = await self.server.convert_gpu_snapshot_bucket.remote(entries, payload, is_last, output_key)
            self.output_batches[output_key] = {
                item["name"]: rebuild_ipc_tensor(item["payload"], self.transport.device) for item in result
            }
            if key is not None:
                self._release_buffer(key)
            return [{k: v for k, v in item.items() if k != "payload"} for item in result]
        except BaseException as error:
            self.transport.failed = f"pipeline IPC conversion failed: {error}"
            raise

    async def store_pipeline_chunks(self, slot, chunks, source, key):
        if source is not None:
            return await self.store_chunks(slot, chunks, source)
        occupied = sum(t.nbytes for values in self.slots.values() for t in values.values())
        required = sum(
            c.numel * torch.empty((), dtype=getattr(torch, c.dtype.removeprefix("torch."))).element_size()
            for c in chunks
        )
        if occupied + required > self.config.max_bytes_per_gpu or any(c.key in self.slots[slot] for c in chunks):
            raise ValueError("duplicate chunk or storage budget exceeded")
        for c in chunks:
            self.slots[slot][c.key] = (
                self.output_batches[key][c.name].reshape(-1)[c.offset : c.offset + c.numel].clone()
            )
        torch.cuda.current_stream().synchronize()

    async def send_pipeline_outputs(self, chunks, destination, key):
        await self.transport.transfer_many(
            [
                (self.output_batches[key][c.name].reshape(-1)[c.offset : c.offset + c.numel], destination, False)
                for c in chunks
            ]
        )

    async def release_pipeline_outputs(self, key):
        if self.transport.failed:
            raise RuntimeError("failed DMA must retain output owners")
        await self.server.release_gpu_snapshot_outputs.remote(key)
        self.output_batches.pop(key)

    async def close(self):
        if self.input_transport.failed or self.output_batches:
            raise RuntimeError("pipeline input failed or outputs remain; retain storage and restart")
        await self.input_transport.close()
        await super().close()


async def export_pipeline(worker, version, publisher, weights, dtype, config):
    """Multi-sender broadcast with deterministic ordering against FSDP collectives.

    Every actor drains the same generator. Relays on rank-zero's node participate
    in each input broadcast. Each broadcast finishes on-device before any actor
    advances to the next FSDP collective, while conversion/distribution run ahead
    independently on the rollout workers.
    """
    source = torch.distributed.get_rank() == 0
    sender = getattr(worker, "_gpu_snapshot_sender", None)
    failure = None
    buffers = [None, None]
    descriptors, used, sequence = [], 0, 0
    metrics = dict(buckets=0, source_bytes=0, broadcast_s=0, input_backpressure_s=0)
    started = time.perf_counter()

    async def flush():
        nonlocal descriptors, used, sequence
        if not descriptors:
            return
        if source:
            start = time.perf_counter()
            await settle([publisher.prepare_gpu_bucket.remote(version, descriptors)])
            metrics["input_backpressure_s"] += time.perf_counter() - start
        if sender is not None:
            index = sequence % 2
            if buffers[index] is None or buffers[index].numel() < used:
                buffers[index] = None
                other = buffers[1 - index]
                if other is not None and other.numel() + used > config.max_tensor_bytes:
                    buffers[1 - index] = None
                buffers[index] = torch.empty(used, dtype=torch.uint8, device="cuda")
            # Do not overlap actor-group collectives with a differently ordered
            # multi-sender collective (the same invariant as the baseline).
            torch.cuda.synchronize()
            start = time.perf_counter()
            await sender.broadcast(buffers[index][:used])
            metrics["broadcast_s"] += time.perf_counter() - start
        if source:
            await settle([publisher.finish_gpu_input.remote()])
        metrics["buckets"] += 1
        metrics["source_bytes"] += used
        sequence += 1
        descriptors, used = [], 0

    try:
        for name, tensor in weights:
            if failure is not None:
                continue
            try:
                target_dtype = dtype if tensor.is_floating_point() else tensor.dtype
                size = tensor.numel() * torch.empty((), dtype=target_dtype).element_size()
                if tensor.device.type != "cuda" or size > config.max_tensor_bytes:
                    raise ValueError("source tensor must fit GPU working budget")
                offset = (used + 15) // 16 * 16
                if descriptors and offset + size > config.bucket_bytes:
                    await flush()
                    offset = 0
                index = sequence % 2
                if source:
                    capacity = max(config.bucket_bytes, size)
                    if buffers[index] is None or buffers[index].numel() < capacity:
                        buffers[index] = None
                        other = buffers[1 - index]
                        if other is not None and other.numel() + capacity > config.max_tensor_bytes:
                            buffers[1 - index] = None  # prior broadcast is complete
                        buffers[index] = torch.empty(capacity, dtype=torch.uint8, device=tensor.device)
                    value = tensor.detach().to(dtype=target_dtype).contiguous()
                    buffers[index][offset : offset + size].copy_(value.reshape(-1).view(torch.uint8))
                    del value
                used = offset + size
                descriptors.append((name, tuple(tensor.shape), str(target_dtype)))
            except BaseException as error:
                failure = error
        if failure is None:
            await flush()
    except BaseException as error:
        failure = failure or error
    if failure is not None:
        if source:
            await settle([publisher.fail_gpu_snapshot.remote(f"pipeline export failed: {failure}")])
        raise failure
    metrics["source_export_s"] = time.perf_counter() - started
    return metrics if source else None
