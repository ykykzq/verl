"""Directed GPU transfers. No tensor payload enters Ray's object store."""

import asyncio
import math
import time

import torch

from verl.checkpoint_engine.gpu_snapshot import tensor_dtype


class DirectedNCCL:
    def __init__(self, timeout_s):
        self.timeout_s = timeout_s
        self.comm = None
        self.failed = None
        self.retained = []
        self.tasks = set()
        self.device = torch.cuda.current_device()

    @staticmethod
    def unique_id():
        from cupy.cuda import nccl

        return nccl.get_unique_id()

    async def _native(self, function, *args):
        if self.failed:
            raise RuntimeError(f"GPU snapshot transport requires worker restart: {self.failed}")

        def run():
            with torch.cuda.device(self.device):
                return function(*args)

        task = asyncio.create_task(asyncio.to_thread(run))
        self.tasks.add(task)
        try:
            result = await asyncio.wait_for(asyncio.shield(task), self.timeout_s)
        except BaseException as error:
            # Cancellation cannot stop native NCCL. Keep tasks, communicator and
            # tensor owners alive; this transport must never be reused afterwards.
            self.failed = f"{type(error).__name__}: {error}"
            self.retained.append(args)
            raise
        else:
            self.tasks.discard(task)
            return result

    async def initialize(self, world_size, unique_id, rank):
        if self.failed:
            raise RuntimeError(f"GPU snapshot transport requires worker restart: {self.failed}")
        if self.comm is not None or self.tasks:
            raise RuntimeError("GPU snapshot communicator is already initialized or has pending native work")

        def initialize():
            from cupy.cuda import nccl

            self.comm = nccl.NcclCommunicator(world_size, unique_id, rank)
            self.stream = torch.cuda.Stream(device=self.device)

        await self._native(initialize)

    async def close(self):
        if self.failed:
            raise RuntimeError(f"GPU snapshot transport requires worker restart: {self.failed}")
        if self.tasks:
            raise RuntimeError("GPU snapshot native operations must settle before communicator close")
        if self.comm is not None:
            await self._native(self.comm.destroy)
            self.comm = None
            self.stream = None

    async def transfer(self, tensor, peer, receive=False):
        if tensor.device.type != "cuda" or not tensor.is_contiguous():
            raise ValueError("NCCL snapshot payload must be contiguous CUDA storage")
        # Record on the producing stream before handing storage to the worker
        # thread; its thread-local default stream need not be the producer's.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))

        def transfer(tensor, ready):
            from cupy.cuda import nccl

            self.stream.wait_event(ready)
            operation = self.comm.recv if receive else self.comm.send
            operation(tensor.data_ptr(), tensor.nbytes, nccl.NCCL_UINT8, peer, self.stream.cuda_stream)
            completed = torch.cuda.Event()
            completed.record(self.stream)
            deadline = time.monotonic() + self.timeout_s
            while not completed.query():
                self.comm.check_async_error()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "NCCL GPU snapshot transfer did not complete; retain buffers and restart workers"
                    )
                time.sleep(0.002)

        await self._native(transfer, tensor, ready)


class GPUSnapshotWorker:
    """One storage participant on each rollout GPU, colocated with RTP."""

    def __init__(self, adapter, config):
        from verl.utils.device import set_expandable_segments

        set_expandable_segments(False)
        if adapter.use_shm or not adapter._ensure_server_handle():
            raise ValueError("nccl_gpu publication requires local CUDA IPC on every TP rank")
        self.adapter = adapter
        self.server = adapter.server_handle
        self.config = config
        self.transport = DirectedNCCL(config.operation_timeout_s)
        self.slots = {}
        self.buffers = {}
        self.outputs = {}
        self.buffer_released = asyncio.Event()

    async def describe(self):
        return await self.server.describe_gpu_snapshot.remote()

    async def unique_id(self):
        return self.transport.unique_id()

    async def initialize(self, world_size, unique_id, rank):
        await self.transport.initialize(world_size, unique_id, rank)

    async def close(self):
        await self.transport.close()
        self.slots.clear()
        self.buffers.clear()
        self.outputs.clear()
        self.buffer_released.set()
        with torch.cuda.device(self.transport.device):
            torch.cuda.empty_cache()

    async def reserve(self, slot):
        if self.transport.failed:
            raise RuntimeError(self.transport.failed)
        self.slots[slot] = {}

    async def begin_conversion(self, version):
        return await self.server.begin_gpu_snapshot.remote(version, self.config.max_conversion_bytes)

    async def allocate(self, key, shape, dtype):
        dtype = tensor_dtype(dtype)
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if size > self.config.max_tensor_bytes:
            raise ValueError(f"GPU tensor requires {size} bytes, above max_tensor_bytes={self.config.max_tensor_bytes}")
        deadline = time.monotonic() + self.config.operation_timeout_s
        while True:
            if self.transport.failed:
                raise RuntimeError(f"GPU snapshot transport requires worker restart: {self.transport.failed}")
            if key in self.buffers:
                raise ValueError(f"GPU buffer already exists: {key}")
            occupied = sum(tensor.nbytes for tensor in self.buffers.values())
            if occupied + size <= self.config.max_tensor_bytes:
                break
            self.buffer_released.clear()
            try:
                await asyncio.wait_for(self.buffer_released.wait(), max(0, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"GPU working buffer budget is busy: occupied={occupied}, requested={size}, "
                    f"max_tensor_bytes={self.config.max_tensor_bytes}"
                ) from None
        self.buffers[key] = torch.empty(shape, dtype=dtype, device="cuda")

    def _release_buffer(self, key):
        if not self.transport.failed:
            self.buffers.pop(key, None)
            self.buffer_released.set()

    async def receive_input(self):
        await self.transport.transfer(self.buffers["input"], 0, receive=True)

    async def convert(self, name=None, is_last=False):
        from verl.workers.rollout.rtp_llm_rollout.ray_weight_transfer import rebuild_ipc_tensor, serialize_ipc_tensor

        tensor = self.buffers.get("input")
        payload = serialize_ipc_tensor(tensor) if tensor is not None else None
        result = await self.server.convert_gpu_snapshot.remote(name, payload, is_last)
        self.outputs = {entry["name"]: rebuild_ipc_tensor(entry["payload"], self.transport.device) for entry in result}
        self._release_buffer("input")
        return [{k: v for k, v in item.items() if k != "payload"} for item in result]

    async def release_outputs(self):
        self.outputs.clear()
        await self.server.release_gpu_snapshot_outputs.remote()

    async def finish_conversion(self):
        await self.server.close_gpu_snapshot.remote()

    async def store_chunk(self, slot, chunk, source):
        if chunk.key in self.slots[slot]:
            raise ValueError(f"duplicate GPU snapshot chunk {chunk.key}")
        dtype = tensor_dtype(chunk.dtype)
        size = chunk.numel * torch.empty((), dtype=dtype).element_size()
        occupied = sum(t.nbytes for chunks in self.slots.values() for t in chunks.values())
        if occupied + size > self.config.max_bytes_per_gpu:
            raise ValueError("GPU snapshot storage capacity exceeded")
        if source is None:
            value = self.outputs[chunk.name].reshape(-1)[chunk.offset : chunk.offset + chunk.numel].clone()
            torch.cuda.current_stream().synchronize()
        else:
            value = torch.empty(chunk.numel, dtype=dtype, device="cuda")
        self.slots[slot][chunk.key] = value
        if source is not None:
            await self.transport.transfer(value, source, receive=True)

    async def send_output(self, chunk, destination):
        value = self.outputs[chunk.name].reshape(-1)[chunk.offset : chunk.offset + chunk.numel]
        await self.transport.transfer(value, destination)

    async def send_chunk(self, slot, chunk, destination):
        await self.transport.transfer(self.slots[slot][chunk.key], destination)

    async def receive_chunk(self, key, slot, chunk, source):
        value = self.buffers[key].reshape(-1)[chunk.offset : chunk.offset + chunk.numel]
        if source is None:
            value.copy_(self.slots[slot][chunk.key])
            torch.cuda.current_stream().synchronize()
        else:
            await self.transport.transfer(value, source, receive=True)

    async def install(self, publisher, snapshot, worker_index, tensors):
        async def weights():
            for name, shape, dtype in tensors:
                key = f"install:{snapshot.version}:{name}"
                await publisher.gpu_fetch_tensor.remote(snapshot, worker_index, name, key)
                try:
                    yield name, self.buffers[key]
                finally:
                    self._release_buffer(key)

        iterator = weights()
        try:
            await self.adapter.update_weights(iterator, global_steps=snapshot.version, wire_format="runtime_tensors")
        except BaseException as error:
            # An IPC consumer may still read its source after a remote failure or cancellation.
            self.transport.failed = f"runtime installation failed: {type(error).__name__}: {error}"
            self.buffer_released.set()
            raise
        finally:
            await iterator.aclose()
