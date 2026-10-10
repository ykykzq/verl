# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator

import ray
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.import_utils import import_external_libs
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, FineGrainedWeightUpdateConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout import BaseRollout, RolloutReplica, get_rollout_class
from verl.workers.rollout.utils import ensure_async_iterator


@dataclass
class TensorMeta:
    name: str
    """The name of the weight tensor."""
    shape: torch.Size
    """The shape of the weight tensor."""
    dtype: torch.dtype
    """The dtype of the weight tensor."""
    chunk_offset: int
    """The chunk offset of the weight tensor."""
    chunk_size: int
    """The chunk size of the weight tensor."""
    offset: int
    """The offset of the weight tensor in the bucket."""


class CheckpointEngineRegistry:
    """Checkpoint engine registry."""

    _registry: dict[str, type["CheckpointEngine"]] = {}

    # Engine modules whose import failed, keyed by module name. Each engine pulls
    # its own transport dependency (cupy for nccl/nixl, nixl, torch_npu for hccl,
    # ...) and `verl.checkpoint_engine` imports them all optionally, so a missing
    # dependency would otherwise only show up as an unregistered backend.
    _import_errors: dict[str, ImportError] = {}

    def register(backend: str):
        """Register a checkpoint engine.

        Args:
            backend: The backend of the checkpoint engine.
        """

        def wrapper(cls: type["CheckpointEngine"]):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls

        return wrapper

    @classmethod
    def record_import_error(cls, module: str, error: ImportError):
        """Record an engine module that could not be imported.

        Args:
            module: The name of the checkpoint engine module.
            error: The import error raised by the module.
        """
        cls._import_errors[module] = error

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        """Get the checkpoint engine class.

        Args:
            backend: The backend of the checkpoint engine.

        Returns:
            The checkpoint engine class.
        """
        if backend not in cls._registry:
            message = f"Checkpoint engine {backend} not registered, registered backends: {sorted(cls._registry)}"
            if cls._import_errors:
                unavailable = ", ".join(f"{module}: {error}" for module, error in sorted(cls._import_errors.items()))
                message += f". Engine modules that failed to import: {unavailable}"
            raise ValueError(message)
        return cls._registry[backend]

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        """Create a new checkpoint engine instance.

        Args:
            backend: The backend of the checkpoint engine.
            *args: Variable length argument pass to the checkpoint engine constructor.
            **kwargs: Arbitrary keyword arguments pass to the checkpoint engine constructor.

        Returns:
            A new checkpoint engine instance.
        """
        return cls.get(backend)(*args, **kwargs)


class CheckpointEngine(ABC):
    """CheckpointEngine is an abstraction to transfer weights from actor to rollout.

    In actor process:
    >>> actor = EngineRegistry.new(...) # FSDP, Megatron, VeOmini, TorchTitan, ...
    >>> engine = CheckpointEngine.new(...) # NCCLCheckpointEngine, NIXLCheckpointEngine, ...
    >>> await engine.send_weights(actor.get_per_tensor_param())

    In rollout process:
    >>> engine = CheckpointEngine.new(...)
    >>> server_adapter = ServerAdapter()
    >>> await server_adapter.update_weights(engine.get_weights()) # update weights via cuda ipc
    """

    # How receive_weights yields weights to the server adapter:
    #   "named_tensors" -- (name, tensor) pairs, bucketed into full-tensor loads.
    #   "delta_flush"   -- (named_tensors, is_last) flushes; the seed may be dense,
    #                       while steady updates carry sparse patches.
    wire_format = "named_tensors"

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Prepare checkpoint engine before each step send_weights/receive_weights.

        1. Allocate weight bucket.
        2. [Optional] Register weight bucket for RDMA.
        3. Return metadata to build communication topology: master ip:port, register RDMA description, etc.

        Args:
            worker_group: The worker group that the checkpoint engine will be used.

        Returns:
            A dictionary that contains the metadata of the worker group.
        """
        raise NotImplementedError

    def prepare_temporary(self) -> dict[str, Any]:
        """Prepare a group whose membership may change on every update.

        Backends opt in by overriding this method and guaranteeing that finalize
        tears down the group, including after partial prepare or initialization.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support temporary replica process groups")

    @classmethod
    @abstractmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build communication topology between all workers.

        Args:
            actor_wg_world_size: The world size of the actor worker group.
            rollout_world_size: The world size of the rollout replica.
            metadata: A list of metadata `prepare` from all workers.

        Returns:
            A tuple of two dictionaries that contains the communication topology for actor and rollout worker group.
            Each dict value should be a list argument equal to the world size of the worker group to dispatch to
            `init_process_group`.

            ```
            world_size = rollout.world_size + actor_wg.world_size
            kwargs = {
                "rank": list(range(world_size)),
                "world_size": [world_size] * world_size,
                "master_metadata": [metadata[0]] * world_size,
            }
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def init_process_group(self, **kwargs):
        """Init process group for checkpoint engine.

        Args:
            **kwargs: Keyword arguments from `build_topology`.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """Finalize checkpoint engine after each step send_weights/receive_weights.

        1. Free weight bucket.
        1. [Optional] Deregister weight bucket for RDMA.
        2. [Optional] Destroy process group.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
            global_steps: Optional trainer step/version associated with this weight update.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Args:
            global_steps: Optional trainer step/version associated with this weight update.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


class CheckpointEngineWithCache(CheckpointEngine):
    """Checkpoint engine with local cache: shm, disk, etc. This allow to synchronize weights without interrupting
    rollout ongoing requests (partial rollout). After requests exhausted, rollout can get weights from local cache.

    Laminar: https://arxiv.org/abs/2510.12633
    """

    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get the weights of the model from local cache.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    """Checkpoint engine for actor and rollout colocated on same GPU.

    In actor process:
    >>> engine = ColocatedCheckpointEngine()
    >>> actor = Actor()
    >>> server_adapter = ServerAdapter()
    >>> engine.send_weights(actor.get_per_tensor_param())
    >>> server_adapter.update_weights(engine.receive_weights())
    """

    def __init__(self, bucket_size: int, is_master: bool = False) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master

    def prepare(self):
        raise NotImplementedError

    def init_process_group(self, **kwargs):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    @classmethod
    def build_topology(cls, *args, **kwargs):
        raise NotImplementedError

    def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
            global_steps: Optional trainer step/version associated with this weight update.
        """
        self.weights = weights

    def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Args:
            global_steps: Optional trainer step/version associated with this weight update.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        yield from self.weights
        self.weights = None


class CheckpointEngineWorker(Worker):
    """CheckpointEngineWorker colocated with inference engine's WorkerProc on same GPU.

    Args:
        rollout_config: The rollout configuration.
        model_config: The model configuration.
        server_adapter: The server adapter to update weights.
    """

    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        server_adapter: BaseRollout = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rollout_config = rollout_config
        self.model_config = model_config

        self.server_adapter: BaseRollout = server_adapter
        backend = self.rollout_config.checkpoint_engine.backend
        if backend == "delta_sharded" and self.rollout_config.name not in {"sglang", "vllm"}:
            raise NotImplementedError(
                f"checkpoint_engine.backend={backend!r} has no delta weight consumer for "
                f"rollout.name={self.rollout_config.name!r}; use sglang or vllm"
            )
        bucket_size = self.rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
        engine_kwargs = self.rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})
        # If custom_backend_module is set, import it so plugins can register
        # in CheckpointEngineRegistry before the backend is instantiated.
        import_external_libs(self.rollout_config.checkpoint_engine.custom_backend_module or None)
        self.checkpoint_engine: CheckpointEngine = CheckpointEngineRegistry.new(
            backend, bucket_size=bucket_size, **engine_kwargs
        )
        self.extra_rollout_args = args
        self.extra_rollout_kwargs = kwargs
        if self.server_adapter is None:
            self.server_adapter = get_rollout_class(self.rollout_config.name, self.rollout_config.mode)(
                *self.extra_rollout_args,
                config=self.rollout_config,
                model_config=self.model_config,
                device_mesh=None,
                **self.extra_rollout_kwargs,
            )
        # sglang and trt-llm need device_mesh for internal communication
        initialize_global_process_group_ray(timeout_second=None, backend="cpu:gloo")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
        await self.server_adapter.update_weights(
            weights,
            global_steps=global_steps,
            wire_format=getattr(self.checkpoint_engine, "wire_format", "named_tensors"),
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_replica_rank(self) -> int:
        """Get replica rank from the underlying rollout server adapter."""
        return self.server_adapter.replica_rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def is_leader_rank(self) -> bool:
        """Get leader rank flag from the underlying rollout server adapter."""
        return self.server_adapter.is_leader_rank


_worker_cls = ray.remote(CheckpointEngineWorker)


class CheckpointEngineManager:
    """Checkpoint engine manager to coordinate weight synchronization between actor and rollout replicas.

    - ME: model engine, FSDP, MCore, VeOmni, export full tensor generator `get_per_tensor_param`
    - CE: checkpoint engine, NCCL, NIXL, etc

    In actor, model engine and checkpoint engine are in same process.
    In rollout, checkpoint engine and rollout worker are in separate process, update weights via cuda ipc.

    ```
    ┌────────┬────────┬─────┬────────┐         ┌───────────────────┬───────────────────┐
    │ ┌────┐ │ ┌────┐ │     │ ┌────┐ │         │     Replica 0     │     Replica 1     │
    │ │ ME0│ │ │ ME1│ │     │ │ MEn│ │         ├────┬────┬────┬────┼────┬────┬────┬────┤
    │ └──┬─┘ │ └────┘ │ ... │ └────┘ │         │ 0  │ 1  │ 2  │ 3  │ 0  │ 1  │ 2  │ 3  │
    │    v   |        |     |        |         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
    | ┌──┴─┐ │ ┌────┐ │     │ ┌────┐ │            ^    ^    ^   cuda ipc   ^    ^    ^
    │ │ CE │ │ │ CE │ │     │ │ CE │ │         ┌──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┐
    │ └──┬─┘ │ └────┘ │     │ └────┘ │         │ CE │ CE │ CE │ CE │ CE │ CE │ CE │ CE |
    └────┼───┴────────┴─────┴────────┘         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
         v                                        |    |    |    |    |    |    |    |
         └─────────────(nccl/nixl/..)─────────────┴────┴────┴────┴────┴────┴────┴────┘
    ```

    Args:
        config: The checkpoint engine config.
        actor_wg: The actor worker group (the training side that produces weights).
        replicas: The list of rollout replicas.
    """

    def __init__(
        self,
        config: CheckpointEngineConfig,
        actor_wg: RayWorkerGroup,
        replicas: list[RolloutReplica],
        fine_grained_config: FineGrainedWeightUpdateConfig | None = None,
        load_balancer_handle: Any = None,
    ) -> None:
        self.fine_grained_config = fine_grained_config or FineGrainedWeightUpdateConfig()
        self.load_balancer = load_balancer_handle
        if self.fine_grained_config.enabled:
            if self.load_balancer is None:
                raise ValueError("fine-grained weight updates require a Router load_balancer_handle")
            if config.backend == "naive":
                raise ValueError("fine-grained weight updates require a standalone checkpoint engine")
        self._weight_update_lock = asyncio.Lock()
        self.config = config
        self.backend = config.backend
        import_external_libs(self.config.custom_backend_module or None)
        self.backend_cls = CheckpointEngineRegistry.get(config.backend)
        if self.fine_grained_config.enabled and (
            getattr(self.backend_cls, "prepare_temporary", CheckpointEngine.prepare_temporary)
            is CheckpointEngine.prepare_temporary
        ):
            raise ValueError(f"checkpoint engine {self.backend!r} does not support temporary replica process groups")
        self.actor_wg = actor_wg
        self.replicas = replicas

    def _require_fine_grained(self) -> None:
        if not self.fine_grained_config.enabled or self.load_balancer is None:
            raise ValueError("fine-grained weight updates must be enabled with a Router")

    def _replica(self, replica_rank: int) -> RolloutReplica:
        for replica in self.replicas:
            if replica.replica_rank == replica_rank:
                return replica
        raise ValueError(f"unknown replica rank {replica_rank}")

    @auto_await
    async def get_replica_states(self) -> dict[int, dict]:
        self._require_fine_grained()
        states = await self.load_balancer.get_replica_states.remote()
        return {r.replica_rank: states[r.server_address] for r in self.replicas}

    @auto_await
    async def pin_replica(self, replica_rank: int, reason: str) -> dict:
        self._require_fine_grained()
        return await self.load_balancer.pin_replica.remote(self._replica(replica_rank).server_address, reason=reason)

    @auto_await
    async def unpin_replica(self, replica_rank: int) -> dict:
        self._require_fine_grained()
        return await self.load_balancer.unpin_replica.remote(self._replica(replica_rank).server_address)

    @auto_await
    async def update_replica_weights(self, replica_rank: int, global_steps: int) -> dict:
        self._require_fine_grained()
        return await self.update_weights(global_steps=global_steps, replica_ranks=[replica_rank])

    @auto_await
    async def recover_replica(self, replica_rank: int, global_steps: int) -> dict:
        self._require_fine_grained()
        if global_steps is None:
            raise ValueError("fine-grained weight updates require an explicit global_steps version")
        async with self._weight_update_lock:
            return await self._update_selected_replicas([self._replica(replica_rank)], global_steps, recovery=True)

    @staticmethod
    async def _run_worker_calls(*calls) -> list:
        """Settle every dispatched RPC before finalization, including on cancellation."""
        refs, errors = [], []
        for call in calls:
            try:
                refs.extend(call())
            except Exception as error:
                errors.append(error)
        pending = asyncio.gather(*refs, return_exceptions=True)
        try:
            results = await asyncio.shield(pending)
        except asyncio.CancelledError:
            await pending
            raise
        errors.extend(result for result in results if isinstance(result, BaseException))
        if errors:
            raise errors[0]
        return results

    async def _transfer_replica_weights(self, replica: RolloutReplica, global_steps: int) -> dict:
        rollout = RayWorkerGroup(
            worker_handles=replica.workers, ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls)
        )
        actor = self.actor_wg
        try:
            metadata = await self._run_worker_calls(
                lambda: actor.execute_checkpoint_engine(["prepare_temporary"] * actor.world_size),
                lambda: rollout.execute_checkpoint_engine(["prepare_temporary"] * rollout.world_size),
            )
            actor_kwargs, rollout_kwargs = self.backend_cls.build_topology(
                actor.world_size, rollout.world_size, metadata
            )
            for group, kwargs in ((actor, actor_kwargs), (rollout, rollout_kwargs)):
                for key, values in kwargs.items():
                    if len(values) != group.world_size:
                        raise ValueError(f"topology {key} must have length {group.world_size}")
                kwargs["method"] = ["init_process_group"] * group.world_size
            await self._run_worker_calls(
                lambda: actor.execute_checkpoint_engine(**actor_kwargs),
                lambda: rollout.execute_checkpoint_engine(**rollout_kwargs),
            )
            results = await self._run_worker_calls(
                lambda: actor.update_weights(global_steps=global_steps, mode=self.backend),
                lambda: rollout.update_weights(global_steps=global_steps),
            )
            metrics = {}
            for result in results[: actor.world_size]:
                if isinstance(result, dict):
                    metrics.update(result)
            return metrics
        finally:
            # Prepare may have partially succeeded before raising. Both sides must
            # finalize even when topology creation, init, or the full stream fails.
            await self._run_worker_calls(
                lambda: actor.execute_checkpoint_engine(["finalize"] * actor.world_size),
                lambda: rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size),
            )

    async def _close_failed_replica(self, replica: RolloutReplica) -> list[BaseException]:
        errors = []
        try:
            await replica.abort_all_requests(reject_request=True)
        except Exception as error:
            errors.append(error)
        calls = []
        for server in replica.servers:
            try:
                calls.append(server.abort_weight_update_from_ipc.remote())
            except Exception as error:
                errors.append(error)
        results = await asyncio.gather(*calls, return_exceptions=True)
        errors.extend(result for result in results if isinstance(result, BaseException))
        return errors

    async def _quarantine_replica(self, replica: RolloutReplica, error: BaseException) -> str:
        async def quarantine():
            cleanup_errors = await self._close_failed_replica(replica)
            detail = f"{type(error).__name__}: {error}"
            if cleanup_errors:
                detail += f"; cleanup errors: {cleanup_errors}"
            await self.load_balancer.fail_replica_update.remote(replica.server_address, detail)
            return detail

        # Keep the transfer lock until isolation completes, even on cancellation.
        cleanup = asyncio.create_task(quarantine())
        try:
            return await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _update_one_replica(self, replica: RolloutReplica, global_steps: int, recovery: bool) -> dict:
        server_id = replica.server_address
        begin = self.load_balancer.begin_replica_recovery if recovery else self.load_balancer.begin_replica_update
        begin_call = asyncio.ensure_future(begin.remote(server_id, global_steps))
        try:
            state = await asyncio.shield(begin_call)
        except asyncio.CancelledError as error:
            # A cancelled wait does not retract a Ray actor state transition.
            # Resolve its outcome before deciding whether cleanup is required.
            try:
                await begin_call
            except Exception:
                raise error from None
            await self._quarantine_replica(replica, error)
            raise
        except Exception:
            # The begin operation checks pin and lifecycle atomically. A pin may
            # have arrived after the caller selected this replica.
            state = (await self.load_balancer.get_replica_states.remote())[server_id]
            if state["pin"] is not None and not recovery:
                return {"status": "pinned", "version": state["weight_version"], "attempts": 0}
            raise
        committed_version = state["weight_version"]
        start = time.monotonic()
        attempts = 0
        published = False
        try:
            while True:
                attempts += 1
                try:
                    await replica.abort_all_requests(reject_request=True)
                    await replica.release_kv_cache()
                    metrics = await self._transfer_replica_weights(replica, global_steps)
                    await replica.resume_kv_cache()
                    await replica.resume_generation()
                    capabilities = await replica.server_handle.get_trajectory_migration_capabilities.remote()
                    if capabilities.get("weight_version") != global_steps:
                        raise RuntimeError("replica did not install the requested weight version")
                    commit_call = asyncio.ensure_future(
                        self.load_balancer.commit_replica_update.remote(server_id, global_steps, metadata=capabilities)
                    )
                    try:
                        await asyncio.shield(commit_call)
                    except asyncio.CancelledError:
                        await commit_call
                        published = True
                        raise
                    published = True
                    return {
                        "status": "updated",
                        "version": global_steps,
                        "attempts": attempts,
                        "duration_s": time.monotonic() - start,
                        "metrics": metrics,
                    }
                except Exception:
                    if attempts > self.fine_grained_config.max_retries:
                        raise
                    cleanup_errors = await self._close_failed_replica(replica)
                    if cleanup_errors:
                        raise RuntimeError(f"replica update cleanup failed: {cleanup_errors}") from cleanup_errors[0]
                    await asyncio.sleep(self.fine_grained_config.retry_backoff_s)
        except BaseException as error:
            if published:
                raise
            detail = await self._quarantine_replica(replica, error)
            if not isinstance(error, Exception) or not self.fine_grained_config.continue_on_failure:
                raise
            return {
                "status": "quarantined",
                "version": committed_version,
                "attempts": attempts,
                "duration_s": time.monotonic() - start,
                "error": detail,
            }

    async def _update_selected_replicas(
        self, replicas: list[RolloutReplica], global_steps: int, recovery: bool = False
    ) -> dict:
        result = {"replicas": {}, "updated": 0, "pinned": 0, "quarantined": 0, "retried": 0}
        for replica in replicas:
            state = (await self.load_balancer.get_replica_states.remote())[replica.server_address]
            if state["lifecycle_state"] == "QUARANTINED" and not recovery:
                entry = {
                    "status": "quarantined",
                    "version": state["weight_version"],
                    "attempts": 0,
                    "error": state["last_update_error"],
                }
            else:
                entry = await self._update_one_replica(replica, global_steps, recovery)
            result["replicas"][replica.replica_rank] = entry
            result[entry["status"]] += 1
            result["retried"] += max(0, entry["attempts"] - 1)
        return result

    def build_process_group(self, rollout: RayWorkerGroup):
        """Build process group for actor worker group and rollout replicas."""
        actor_wg = self.actor_wg

        # 1. prepare all workers
        metadata = ray.get(
            actor_wg.execute_checkpoint_engine(["prepare"] * actor_wg.world_size)
            + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        )

        # 2. build communication topology between all workers
        actor_wg_kwargs, rollout_kwargs = self.backend_cls.build_topology(
            actor_wg.world_size, rollout.world_size, metadata
        )
        for k, v in actor_wg_kwargs.items():
            assert len(v) == actor_wg.world_size, f"actor_wg_kwargs[{k}] must have length of {actor_wg.world_size}"
        for k, v in rollout_kwargs.items():
            assert len(v) == rollout.world_size, f"rollout_kwargs[{k}] must have length of {rollout.world_size}"

        actor_wg_kwargs["method"] = ["init_process_group"] * actor_wg.world_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size

        # 3. init process group between all workers
        ray.get(
            actor_wg.execute_checkpoint_engine(**actor_wg_kwargs) + rollout.execute_checkpoint_engine(**rollout_kwargs)
        )

    def add_replicas(self, replicas: list[RolloutReplica]):
        """Add rollout replicas to the manager for elastic scale up, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to add.
        """
        self.replicas.extend(replicas)

    def remove_replicas(self, replicas: list[RolloutReplica]):
        """Remove rollout replicas from the manager for elastic scale down, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to remove.
        """
        replicas_set = set(replicas)
        self.replicas = [r for r in self.replicas if r not in replicas_set]

    @auto_await
    async def sleep_replicas(self):
        """Sleep all rollout replicas: free weight and kv_cache device memory."""
        await asyncio.gather(*[r.sleep() for r in self.replicas])

    @auto_await
    async def wake_up_replicas(self):
        """Resume all rollout replicas: recover kv_cache and weights device memory."""
        await asyncio.gather(*[r.wake_up() for r in self.replicas])

    @auto_await
    async def abort_replicas(self, reject_request: bool = False):
        """Abort all in-flight requests on every replica.

        Args:
            reject_request: Fail requests arriving behind the closed gate instead of
                parking them, for replicas that will not resume generation soon.
        """
        await asyncio.gather(*[r.abort_all_requests(reject_request=reject_request) for r in self.replicas])

    @auto_await
    async def resume_generation_replicas(self):
        """Resume eligible replicas without reopening isolated weight updates."""
        if self.fine_grained_config.enabled:
            async with self._weight_update_lock:
                states = await self.load_balancer.get_replica_states.remote()
                await asyncio.gather(
                    *[
                        r.resume_generation()
                        for r in self.replicas
                        if states[r.server_address]["lifecycle_state"] == "SERVING"
                    ]
                )
            return
        await asyncio.gather(*[r.resume_generation() for r in self.replicas])

    @auto_await
    async def release_kv_cache_replicas(self):
        """Release kv_cache of all rollout replicas before NCCL weight sync.

        Unlike sleep_replicas(), this only frees the kv_cache and leaves model
        weights untouched, so the NCCL transfer can write directly into the
        existing weight buffers.  Call resume_kv_cache_replicas() after sync.
        """
        await asyncio.gather(*[r.release_kv_cache() for r in self.replicas])

    @auto_await
    async def resume_kv_cache_replicas(self):
        """Restore kv_cache of all rollout replicas after NCCL weight sync.

        Counterpart to release_kv_cache_replicas().
        """
        await asyncio.gather(*[r.resume_kv_cache() for r in self.replicas])

    @auto_await
    async def update_weights(self, global_steps: int = None, replica_ranks: list[int] | None = None):
        """Update weights from actor worker group to rollout replicas.

        Args:
            global_steps: The global steps of the actor worker group.
            replica_ranks: Target ranks for fine-grained updates; omitted updates
                all unpinned, non-quarantined replicas sequentially.
        """

        if self.fine_grained_config.enabled:
            self._require_fine_grained()
            if global_steps is None:
                raise ValueError("fine-grained weight updates require an explicit global_steps version")
            selected = list(self.replicas) if replica_ranks is None else [self._replica(rank) for rank in replica_ranks]
            if len({r.replica_rank for r in selected}) != len(selected):
                raise ValueError("replica_ranks must not contain duplicate ranks")
            async with self._weight_update_lock:
                if replica_ranks is not None:
                    states = await self.load_balancer.get_replica_states.remote()
                    for replica in selected:
                        if states[replica.server_address]["lifecycle_state"] == "QUARANTINED":
                            raise RuntimeError(f"replica {replica.replica_rank} is quarantined; use recover_replica")
                return await self._update_selected_replicas(selected, global_steps)
        if replica_ranks is not None:
            raise ValueError("replica_ranks requires fine-grained weight updates to be enabled")

        # 0. update weights for sync training with colocated actor and rollout
        if self.backend == "naive":
            ray.get(self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend))
            return {}

        # 1. abort and save all unfinished requests for partial rollout
        await self.abort_replicas()

        # 2. create a temporay worker group for all replicas
        workers = []
        for replica in self.replicas:
            workers.extend(replica.workers)
        rollout = RayWorkerGroup(worker_handles=workers, ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls))
        actor_wg = self.actor_wg

        # 3. release kv_cache before weight sync (weights stay in place)
        await self.release_kv_cache_replicas()

        # 4. build process group
        self.build_process_group(rollout)

        # 5. update weights of all workers
        results = ray.get(
            actor_wg.update_weights(global_steps=global_steps, mode=self.backend)
            + rollout.update_weights(global_steps=global_steps)
        )
        # The sender workers return the engine's per-sync metrics (empty for
        # backends that don't track any); merge and hand them to the trainer.
        sync_metrics: dict = {}
        for result in results[: actor_wg.world_size]:
            if isinstance(result, dict):
                sync_metrics.update(result)

        # 6. finalize all workers
        ray.get(
            actor_wg.execute_checkpoint_engine(["finalize"] * actor_wg.world_size)
            + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
        )

        # 7. restore kv_cache after weight sync
        await self.resume_kv_cache_replicas()

        # 8. resume all unfinished requests for partial rollout
        await self.resume_generation_replicas()

        return sync_metrics


async def split_weight_chunks(
    weights: Generator[tuple[str, torch.Tensor], None, None], bucket_size: int, meta_only: bool = False
) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor | None], None]:
    """Split the weight into chunks.

    Args:
        weights: The weights generator.
        bucket_size: Max bucket size in bytes.

    Yields:
        A tuple of the weight chunk metadata and the buffer.
    """
    async for name, weight in ensure_async_iterator(weights):
        buffer = weight.view(-1).view(torch.uint8)
        chunk_offset = 0
        while chunk_offset < weight.nbytes:
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            tensor_meta = TensorMeta(
                name=name,
                shape=weight.shape,
                dtype=weight.dtype,
                chunk_offset=chunk_offset,
                chunk_size=chunk_size,
                offset=None,
            )
            yield (tensor_meta, None if meta_only else buffer[chunk_offset : chunk_offset + chunk_size])
            chunk_offset += chunk_size


async def merge_weight_chunks(
    chunks: Generator[tuple[TensorMeta, torch.Tensor], None, None], bucket_size: int
) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
    """Merge the weight chunks into the original weight.

    Args:
        chunks: The chunks generator.
        bucket_size: Max bucket size in bytes.

    Yields:
        A tuple of the name of the weight tensor and the tensor itself.
    """
    merge_name, merge_weight, merge_buffer, merge_offset = None, None, None, 0
    async for tensor_meta, chunk in chunks:
        assert chunk.dtype == torch.uint8, f"Chunk dtype must be uint8, but got {chunk.dtype}"
        nbytes = tensor_meta.shape.numel() * tensor_meta.dtype.itemsize

        # weight is small enough to fit in one bucket
        if nbytes <= bucket_size:
            assert merge_weight is None, f"Weight must be None, but got {merge_name}"
            name, weight = tensor_meta.name, chunk.view(tensor_meta.dtype).view(tensor_meta.shape)
            yield (name, weight)
            continue

        if merge_weight is None:
            assert tensor_meta.chunk_offset == 0, f"Chunk offset must be 0, but got {tensor_meta}"
            merge_name, merge_weight = (
                tensor_meta.name,
                torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype, device=chunk.device),
            )
            merge_buffer = merge_weight.view(-1).view(torch.uint8)
            merge_offset = 0

        assert tensor_meta.name == merge_name
        assert merge_offset == tensor_meta.chunk_offset
        merge_buffer[tensor_meta.chunk_offset : tensor_meta.chunk_offset + tensor_meta.chunk_size] = chunk
        merge_offset += tensor_meta.chunk_size
        if tensor_meta.chunk_offset + tensor_meta.chunk_size == nbytes:
            yield (merge_name, merge_weight)
            merge_name, merge_weight, merge_buffer, merge_offset = None, None, None, 0
