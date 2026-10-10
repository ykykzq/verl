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
"""
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
"""

import asyncio
import hashlib
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import normalize_token_ids
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tracking import RLInsightLogger
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.router import GlobalRequestLoadBalancer  # noqa: F401
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ROUTE_SERVER_ID_FIELD = "_verl_route_server_id"


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        **kwargs,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry. Optional; subclasses that
                manage server routing externally can pass None.
        """
        self.config = config
        self._load_balancer = load_balancer_handle
        # Balancer field declarations, queried lazily once.
        self._lb_require_acquire_fields: list[str] | None = None
        self._lb_require_release_fields: list[str] | None = None

    def _trajectory_migration_config(self) -> Any | None:
        rollout_config = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
        migration_config = getattr(rollout_config, "trajectory_migration", None)
        if migration_config is None or not bool(getattr(migration_config, "enabled", False)):
            return None
        return migration_config

    def _fine_grained_weight_update_enabled(self) -> bool:
        rollout_config = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
        return bool(getattr(getattr(rollout_config, "fine_grained_weight_update", None), "enabled", False))

    async def _acquire_server(self, request_id: str, **extra) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        # Only the declared fields are serialized.
        if self._lb_require_acquire_fields is None:
            acquire_fields, release_fields = await asyncio.gather(
                self._load_balancer.require_acquire_fields.remote(),
                self._load_balancer.require_release_fields.remote(),
            )
            self._lb_require_acquire_fields = list(acquire_fields)
            self._lb_require_release_fields = list(release_fields)
        fields = {name: extra[name] for name in self._lb_require_acquire_fields if name in extra}
        return await self._load_balancer.acquire_server.remote(request_id=request_id, **fields)

    def _release_server(self, server_id: str, request_id: str | None = None) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        pool = {"request_id": request_id}
        fields = {name: pool[name] for name in self._lb_require_release_fields if name in pool}
        self._load_balancer.release_server.remote(server_id=server_id, **fields)

    def _vllm_request_id(self, request_id: str) -> str:
        # request_id passed to vLLM. Default: a fresh uuid per turn so each turn
        # is an independent vLLM request. Under full_determinism the caller's
        # request_id is passed straight through so vLLM sees a stable id across runs.
        if (
            getattr(self.config.actor_rollout_ref.rollout, "full_determinism", False)
            or self._trajectory_migration_config() is not None
        ):
            return request_id
        return uuid4().hex

    async def abort_trajectory(self, request_id: str) -> dict[str, Any]:
        """Stop one active trajectory on its currently routed replica."""
        if self._trajectory_migration_config() is None:
            return {
                "aborted": False,
                "request_id": request_id,
                "reason": "trajectory migration is disabled",
            }
        return await self._load_balancer.abort_trajectory.remote(request_id)

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output
        """
        server_id, server = await self._acquire_server(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            **kwargs,
        )
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            # SGLang cannot take raw frames: its video_data accepts only a path/url/base64 or a
            # processor_output dict. Always hand it the pre-computed payload (None when the model's
            # processor produced no pixel_values_videos, e.g. a non-Qwen-VL family) and never the raw
            # frames -- dropping video beats sending SGLang tensors it cannot parse. vLLM keeps the
            # frames and never enters this branch. Neither server signature accepts **kwargs, so pop.
            mm_processor_output = kwargs.pop("mm_processor_output", None)
            if self.config.actor_rollout_ref.rollout.name == "sglang":
                video_data = mm_processor_output
            # priority is only supported by vLLM rollout server.
            priority = kwargs.pop("priority", 0)
            priority_kwargs = (
                {"priority": priority} if priority != 0 and self.config.actor_rollout_ref.rollout.name == "vllm" else {}
            )
            output: TokenOutput = await server.generate.remote(
                request_id=self._vllm_request_id(request_id),  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **priority_kwargs,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            if self._trajectory_migration_config() is not None:
                output.extra_fields[_ROUTE_SERVER_ID_FIELD] = server_id
            return output
        finally:
            self._release_server(
                server_id,
                request_id=request_id,
            )


class FullyAsyncLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        only_hybrid: bool = False,
        **kwargs,
    ):
        """Initialize the FullyAsyncLLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry.
            only_hybrid (bool): When ``True``, hybrid replicas are the *only* rollout
                resource.  If the load balancer is temporarily empty (e.g. during
                weight synchronisation) :meth:`_acquire_server` will keep retrying
                every 1 second instead of raising immediately.
        """
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._only_hybrid = only_hybrid

    async def _acquire_server(self, request_id: str, **extra) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        # When only_hybrid is True, hybrid replicas are the sole rollout resource and
        # the LB may be temporarily empty during weight sync / scaling transitions.
        # In that case keep retrying every 1 s until a server becomes available.
        # Fine-grained updates also wait while a registered replica is updating.
        while True:
            try:
                return await super()._acquire_server(request_id, **extra)
            except RuntimeError as e:
                if "No available servers in load balancer" not in str(e):
                    raise
                if self._fine_grained_weight_update_enabled():
                    await self._wait_for_fine_grained_replica()
                    continue
                if not self._only_hybrid:
                    raise
                await asyncio.sleep(1)

    async def _wait_for_fine_grained_replica(self) -> None:
        states = await self._load_balancer.get_replica_states.remote()
        if states and all(state["lifecycle_state"] == "QUARANTINED" for state in states.values()):
            raise RuntimeError("All rollout replicas are quarantined; recover a replica before retrying")
        if not any(state["lifecycle_state"] in ("SERVING", "UPDATING") for state in states.values()):
            raise RuntimeError("No available servers in load balancer")
        await asyncio.sleep(1)

    def _configured_response_length(self) -> Optional[int]:
        """Per-response token budget from the rollout config, or ``None`` when unavailable.

        Tests and lightweight callers may pass a config stub without the rollout section; in that
        case the resume loop keeps its previous behaviour of deferring to the server default.
        """
        rollout_config = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
        response_length = getattr(rollout_config, "response_length", None)
        if isinstance(response_length, int) and response_length > 0:
            return response_length
        return None

    async def _rollback_trajectory_migration(
        self,
        request_id: str,
        decision_id: str,
        target_server: ray.actor.ActorHandle,
        ticket: dict[str, Any] | None,
    ) -> None:
        if ticket is not None and ticket.get("prefix_digest") is not None:
            try:
                await target_server.discard_trajectory_migration.remote(
                    request_id=request_id,
                    prefix_digest=ticket["prefix_digest"],
                )
            except Exception:
                logger.exception("failed to discard target-side trajectory state for %s", request_id)
        try:
            await self._load_balancer.cancel_trajectory_migration.remote(
                request_id=request_id,
                decision_id=decision_id,
            )
        except Exception:
            logger.exception("failed to cancel trajectory migration reservation for %s", request_id)

    async def _try_migrate_trajectory(
        self,
        request_id: str,
        source_server_id: str,
        prompt_ids: list[int],
        final_output: TokenOutput,
        checkpoint_index: int,
        source_weight_version: Any,
        sampling_params: dict[str, Any],
        migration_required: bool = False,
    ) -> dict[str, Any] | None:
        migration_config = self._trajectory_migration_config()
        if migration_config is None or self._load_balancer is None:
            return None

        trajectory_summary = {
            "request_id": request_id,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(final_output.token_ids),
            "prefix_tokens": len(prompt_ids) + len(final_output.token_ids),
            "checkpoint_index": checkpoint_index,
        }
        if migration_required:
            trajectory_summary["migration_required"] = True
        source_metadata = {
            "weight_version": "initial" if source_weight_version is None else source_weight_version,
        }
        decision = await self._load_balancer.plan_trajectory_migration.remote(
            request_id=request_id,
            source_server_id=source_server_id,
            trajectory=trajectory_summary,
            source_metadata=source_metadata,
        )
        if not decision.get("migrate", False):
            return None

        decision_id = decision["decision_id"]
        timeout_s = float(getattr(migration_config, "transfer_timeout_s", 30.0))
        trajectory_state = {
            "request_id": request_id,
            "prompt_ids": prompt_ids,
            "generated_token_ids": list(final_output.token_ids),
            "checkpoint_index": checkpoint_index,
            "sampling_params": dict(sampling_params),
        }
        ticket = None
        try:
            if decision.get("resume_source", False):
                await self._load_balancer.commit_trajectory_migration.remote(
                    request_id=request_id,
                    decision_id=decision_id,
                )
                return {"resume_source": True}
            mode = decision.get("mode", "remote_prefix")
            source_version = decision.get("source_version", source_metadata["weight_version"])
            target_version = decision.get("target_version", decision["target_metadata"].get("weight_version"))
            if mode == "recompute":
                digest = hashlib.sha256()
                for token_id in prompt_ids + final_output.token_ids:
                    digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
                ticket = {
                    "backend": "recompute",
                    "mode": mode,
                    "model_id": decision["source_metadata"]["model_id"],
                    "request_id": request_id,
                    "prefix_tokens": trajectory_summary["prefix_tokens"],
                    "prefix_digest": digest.hexdigest(),
                    "source_version": source_version,
                    "target_version": target_version,
                }
            else:
                ticket = await asyncio.wait_for(
                    decision["source_server"].prepare_trajectory_migration.remote(
                        trajectory_state=trajectory_state,
                        target_metadata=decision["target_metadata"],
                        backend=str(getattr(migration_config, "kv_transfer_backend", "remote_prefix")),
                        timeout_s=timeout_s,
                    ),
                    timeout=timeout_s,
                )
            accepted = await asyncio.wait_for(
                decision["target_server"].accept_trajectory_migration.remote(
                    ticket=ticket,
                    trajectory_state=trajectory_state,
                ),
                timeout=timeout_s,
            )
            if not accepted.get("accepted", False):
                raise RuntimeError(accepted.get("reason", "target rejected trajectory transfer"))
            await self._load_balancer.commit_trajectory_migration.remote(
                request_id=request_id,
                decision_id=decision_id,
            )
            return {
                "source_server_id": source_server_id,
                "target_server_id": decision["target_server_id"],
                "checkpoint_index": checkpoint_index,
                "prefix_tokens": trajectory_summary["prefix_tokens"],
                "backend": ticket["backend"],
                "mode": mode,
                "source_version": source_version,
                "target_version": target_version,
                "migration_ticket": ticket,
            }
        except asyncio.CancelledError:
            await asyncio.shield(
                self._rollback_trajectory_migration(
                    request_id=request_id,
                    decision_id=decision_id,
                    target_server=decision["target_server"],
                    ticket=ticket,
                )
            )
            raise
        except Exception:
            await self._rollback_trajectory_migration(
                request_id=request_id,
                decision_id=decision_id,
                target_server=decision["target_server"],
                ticket=ticket,
            )
            logger.exception(
                "trajectory migration failed; keeping %s on replica %s",
                request_id,
                source_server_id,
            )
            return None

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.
            audio_data (Optional[List[Any]]): Audio data for the chat completion.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Multimodal processor kwargs.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        # The budget below is rewritten on every attempt, and the caller reuses its dict across
        # turns, so never mutate the caller's copy.
        sampling_params = dict(sampling_params)

        migration_config = self._trajectory_migration_config()
        checkpoint_tokens = int(getattr(migration_config, "checkpoint_tokens", 0)) if migration_config else None
        versioned_continuation = migration_config is not None and (
            self._fine_grained_weight_update_enabled()
            or bool(getattr(migration_config, "allow_cross_version_recompute", False))
        )

        if original_max_tokens is None:
            # Without an explicit limit each attempt falls back to the server-side default, which is
            # derived from len(prompt_ids) and is only correct on the first attempt: a resume passes
            # prompt + tokens generated so far, so the default charges generated tokens against the
            # *prompt* budget and re-permits close to a full response_length every time. Pin the
            # cumulative budget here instead, which also makes the bookkeeping in step 3 effective.
            response_length = self._configured_response_length()
            if response_length is not None:
                limit_key = "max_tokens"
                original_max_tokens = response_length
                sampling_params[limit_key] = response_length

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None
        # Prefix-cache hits are reported per prefill. The base client returns the
        # server's TokenOutput directly (so sync mode surfaces num_cached_tokens),
        # but here we rebuild a fresh TokenOutput across resume iterations, so we
        # must carry it forward explicitly or the consumer sees 0. Take the first
        # (initial-prompt) prefill's hit count, matching single-prefill semantics.
        num_cached_tokens = None
        migration_history: list[dict[str, Any]] = []
        migration_replans = 0
        forced_prefill_tokens = None
        migration_ticket = None
        pending_migration = None
        replan_migration = False
        last_source_server_id = None
        last_source_version = None
        checkpoint_index = 0

        while True:
            if replan_migration:
                migration = await self._try_migrate_trajectory(
                    request_id=request_id,
                    source_server_id=last_source_server_id,
                    prompt_ids=prompt_ids,
                    final_output=final_output,
                    checkpoint_index=checkpoint_index,
                    source_weight_version=last_source_version,
                    sampling_params=sampling_params,
                    migration_required=True,
                )
                if migration is None:
                    if self._fine_grained_weight_update_enabled():
                        await self._wait_for_fine_grained_replica()
                    else:
                        await asyncio.sleep(1)
                    continue
                migration_ticket = migration.pop("migration_ticket", None)
                pending_migration = None if migration.get("resume_source", False) else migration
                replan_migration = False

            if migration_config is not None:
                if original_max_tokens is None or limit_key is None:
                    raise ValueError(
                        "trajectory migration requires an explicit max_tokens/max_new_tokens or rollout.response_length"
                    )
                remaining_tokens = original_max_tokens - len(final_output.token_ids)
                sampling_params[limit_key] = min(checkpoint_tokens, remaining_tokens)

            continuation_kwargs = {}
            if versioned_continuation and final_output.token_ids:
                continuation_kwargs["expected_source_version"] = (
                    "initial" if last_source_version is None else last_source_version
                )
            if migration_ticket is not None:
                continuation_kwargs["migration_ticket"] = migration_ticket

            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **continuation_kwargs,
                **kwargs,
            )
            migration_ticket = None
            if migration_config is not None and output.extra_fields.get("migration_rejected", False):
                # A cleared acceptance or a reroute to newer weights must be planned again.
                migration_replans += 1
                pending_migration = None
                replan_migration = versioned_continuation
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                continue
            if pending_migration is not None:
                migration_history.append(pending_migration)
                pending_migration = None

            # 2. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            # On partial rollout resume the model version may differ, so keep
            # existing routing and only append routing for newly generated tokens.
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = np.concatenate(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]]
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            # sampler top-k heads for score centering, one row per newly generated token
            for key in ("response_topk_ids", "response_topk_log_probs"):
                if key in output.extra_fields:
                    previous = final_output.extra_fields.get(key)
                    final_output.extra_fields[key] = (
                        output.extra_fields[key]
                        if previous is None
                        else np.concatenate([previous, output.extra_fields[key]])
                    )
            final_output.stop_reason = output.stop_reason

            # carry the initial prefill's prefix-cache hit count forward
            if num_cached_tokens is None:
                num_cached_tokens = output.extra_fields.get("num_cached_tokens")
            if "forced_prefill_tokens" in output.extra_fields:
                forced_prefill_tokens = (forced_prefill_tokens or 0) + output.extra_fields["forced_prefill_tokens"]

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            source_server_id = output.extra_fields.pop(_ROUTE_SERVER_ID_FIELD, None)
            if output.token_ids:
                last_source_server_id = source_server_id
                last_source_version = global_steps
                if global_steps is not None:
                    min_global_steps = global_steps if min_global_steps is None else min(min_global_steps, global_steps)
                    max_global_steps = global_steps if max_global_steps is None else max(max_global_steps, global_steps)

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            if migration_config is not None and output.extra_fields.get("migration_checkpoint", False):
                checkpoint_index += 1
                if source_server_id is None:
                    raise RuntimeError("rollout server response did not include its routed replica")
                migration = await self._try_migrate_trajectory(
                    request_id=request_id,
                    source_server_id=source_server_id,
                    prompt_ids=prompt_ids,
                    final_output=final_output,
                    checkpoint_index=checkpoint_index,
                    source_weight_version=global_steps,
                    sampling_params=sampling_params,
                )
                if migration is not None:
                    migration_ticket = migration.pop("migration_ticket")
                    pending_migration = migration
                continue

            # 4. check stop reason
            # If partial rollout not enable, aborted samples will be dropped.
            # For v1 trainer, should_retry is always True. Since self.config.async_training is not exist.
            should_retry = True
            if hasattr(self.config, "async_training") and not self.config.async_training.partial_rollout:
                should_retry = False
            if output.stop_reason not in ("aborted", "abort") or not should_retry:
                break

            await asyncio.sleep(1)

        final_output.extra_fields["global_steps"] = last_source_version if final_output.token_ids else global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        final_output.extra_fields["num_cached_tokens"] = num_cached_tokens
        if forced_prefill_tokens is not None:
            final_output.extra_fields["forced_prefill_tokens"] = forced_prefill_tokens
        if migration_config is not None:
            final_output.extra_fields["forced_prefill_tokens"] = forced_prefill_tokens or 0
            final_output.extra_fields["trajectory_migrations"] = migration_history
            final_output.extra_fields["trajectory_migration_counts"] = {
                mode: sum(migration["mode"] == mode for migration in migration_history)
                for mode in ("remote_prefix", "recompute")
            }
            final_output.extra_fields["trajectory_migration_replans"] = migration_replans
        return final_output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
        start_rank (int): First ``replica_rank`` to assign.  Defaults to 0.
        load_balancer_cls: Optional subclass of the default router strategy's
            load balancer to use as the routing actor (wrapped with
            ``ray.remote`` at instantiation). When given, it takes precedence
            over ``rollout.router_config_path``. Pass a subclass that
            overrides :meth:`acquire_server` to take full control of routing.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        start_rank: int = 0,
        load_balancer_cls: type | None = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.start_rank = start_rank
        self._load_balancer_cls = load_balancer_cls

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    @classmethod
    @auto_await
    async def create_empty(cls, *args, **kwargs):
        """Create an empty LLMServerManager: no replicas are launched and the
        load balancer starts empty.

        Used by trainers that serve rollout exclusively from standalone
        resources (e.g. v1 separate_async with ``actor_rollout_ref.hybrid_engine=False``)
        so that no inference engine is initialized on the training GPUs.
        """
        instance = cls(*args, **kwargs)
        instance.rollout_replicas = []
        instance.server_handles = []
        instance.server_addresses = []
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self, start_rank: int = None):
        """Initialize the LLM server replicas.

        Args:
            start_rank: First ``replica_rank`` to assign.  Defaults to ``self.start_rank``
                so standalone replicas can avoid Ray named-actor collisions with hybrid
                replicas (which start at 0) when both coexist (e.g. separate async).
        """
        if start_rank is None:
            start_rank = self.start_rank
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        # PD inflates per-replica footprint; miss this and init_hybrid slices
        # past worker_group → empty workers on replica_rank>=1.
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        self.server_metadata = {}
        migration_config = getattr(self.rollout_config, "trajectory_migration", None)
        fine_grained_config = getattr(self.rollout_config, "fine_grained_weight_update", None)
        if getattr(migration_config, "enabled", False) or getattr(fine_grained_config, "enabled", False):
            for address, handle in zip(self.server_addresses, self.server_handles, strict=True):
                try:
                    self.server_metadata[address] = await handle.get_trajectory_migration_capabilities.remote()
                except AttributeError:
                    self.server_metadata[address] = {}
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus / rl-insight metrics with server addresses
        needs_metrics = self.rollout_config.prometheus.enable or RLInsightLogger.enabled()
        if self.rollout_config.disable_log_stats:
            if needs_metrics:
                raise ValueError("Metrics monitoring requires disable_log_stats=False, but it is currently True.")
        if not self.rollout_config.disable_log_stats:
            if self.rollout_config.prometheus.enable:
                update_prometheus_config(
                    self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name
                )
            if RLInsightLogger.enabled():
                RLInsightLogger.register_rollout_metrics(
                    self.server_addresses,
                    self.rollout_config.name,
                    labels=[{"replica": server.replica_rank} for server in self.rollout_replicas],
                )

    async def _init_global_load_balancer(self) -> None:
        from verl.workers.rollout.router import get_router_handle

        migration_config = getattr(self.rollout_config, "trajectory_migration", None)
        if isinstance(migration_config, DictConfig):
            migration_config = OmegaConf.to_container(migration_config, resolve=True)
        elif is_dataclass(migration_config):
            migration_config = asdict(migration_config)
        elif migration_config is not None:
            migration_config = dict(migration_config)
        self.global_load_balancer = get_router_handle(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            router_config_path=getattr(self.rollout_config, "router_config_path", None),
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
            load_balancer_cls=self._load_balancer_cls,
            trajectory_migration_config=migration_config,
            server_metadata=getattr(self, "server_metadata", {}),
        )

    def get_client(self, client_cls: type[LLMServerClient] | None = None, **kwargs) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            client_cls: The client class to instantiate. Defaults to
                :class:`FullyAsyncLLMServerClient` when trajectory migration is enabled,
                otherwise :class:`LLMServerClient`. Pass a subclass to customize
                request-id handling (e.g. a deterministic client that forwards
                the caller's ``request_id`` straight to vLLM), or
                :class:`FullyAsyncLLMServerClient` for abort-resume support.
            **kwargs: Forwarded to the client constructor.
        """
        if client_cls is None:
            migration_config = getattr(self.rollout_config, "trajectory_migration", None)
            client_cls = FullyAsyncLLMServerClient if getattr(migration_config, "enabled", False) else LLMServerClient
        return client_cls(
            config=self.config,
            load_balancer_handle=self.global_load_balancer,
            **kwargs,
        )

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])
