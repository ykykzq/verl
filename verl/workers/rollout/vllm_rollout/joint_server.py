"""Version-fenced Joint request lifecycle for standalone vLLM TP1 replicas."""

import asyncio
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.joint_snapshot import JointSnapshotMixin


def prefix_digest(tokens):
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


class JointVLLMServerMixin(JointSnapshotMixin):
    def _joint_initialize(self):
        self._joint_enabled = bool(self.config.weight_publication.enabled)
        if not self._joint_enabled:
            return
        self._require_hf_snapshot()
        if self._disaggregation_role != "null" or self.node_rank != 0:
            raise ValueError("V17 Joint requires one standalone TP1 engine per replica")
        self._joint_run_id = os.environ["VERL_V17_RUN_ID"]
        self._joint_active = {}
        self._joint_by_engine = {}
        self._joint_exports = {}
        self._joint_accepted = {}
        self._joint_lease_seconds = max(120, math.ceil(2 * self.config.trajectory_migration.transfer_timeout_s))
        os.environ["VERL_V17_JOINT_ENABLE"] = "1"
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = ray_node_ip()
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(48000 + self.replica_rank * 8)
        directory = os.environ["VERL_V17_REPREFILL_PROFILE_DIR"]
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._joint_profile_file = (Path(directory) / f"vllm-server-r{self.replica_rank}-p{os.getpid()}.jsonl").open(
            "x", buffering=1
        )
        self._joint_event("device_binding", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))

    def _joint_version(self):
        return "initial" if self.global_steps is None else self.global_steps

    def _joint_event(self, event, **fields):
        self._joint_profile_file.write(
            json.dumps(
                dict(
                    event=event,
                    wall_time=time.time(),
                    replica_rank=self.replica_rank,
                    weight_version=self._joint_version(),
                    **fields,
                ),
                separators=(",", ":"),
            )
            + "\n"
        )

    def _joint_engine_options(self, args):
        if not self._joint_enabled:
            return
        import torch
        from verl.checkpoint_engine.snapshot_memory import snapshot_memory_role

        cfg = self.config.weight_publication.gpu_snapshot
        role = snapshot_memory_role(
            cfg, self.replica_rank, self.config.checkpoint_engine.update_weights_bucket_megabytes << 20
        )
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = float(args["gpu_memory_utilization"]) - role.total_bytes / total
        if not 0 < fraction < 1:
            raise MemoryError("vLLM budget cannot accommodate the reserved immutable snapshot storage")
        args["gpu_memory_utilization"] = fraction
        args["kv_transfer_config"] = json.dumps(
            {
                "kv_connector": "JointNixlConnector",
                "kv_role": "kv_both",
                "kv_buffer_device": "cuda",
                "kv_connector_module_path": "verl.workers.rollout.vllm_rollout.joint_nixl_connector",
                "engine_id": f"{self._joint_run_id}-r{self.replica_rank}-{uuid.uuid4().hex}",
                "kv_connector_extra_config": {"kv_lease_duration": self._joint_lease_seconds},
            }
        )
        self._joint_event("engine_budget", memory_role=role.as_dict(), effective_gpu_memory_utilization=fraction)

    async def _joint_native(self, method, *args):
        return await self.engine.engine_core.call_utility_async(method, *args)

    async def get_laminar_kv_usage(self):
        return await self._joint_native("verl_kv_cache_usage")

    async def get_trajectory_migration_capabilities(self):
        native = await self._joint_native("v17_scheduler_state")
        version = self._joint_version()
        identity = self.model_config.local_path
        return dict(
            backend="vllm",
            replica_rank=self.replica_rank,
            model_id=identity,
            weight_version=version,
            rollout_config_fingerprint=f"v17-qwen2-tp1-{self.config.dtype}",
            kv_block_size_tokens=native["kv_block_size_tokens"],
            max_sequence_tokens=native["max_model_len"],
            max_batch_size=native["max_num_seqs"],
            native_initialized_kv_limit=native["max_num_seqs"],
            kv_capacity_tokens=native["kv_capacity_tokens"],
            kv_transfer_backends=["remote_prefix"],
            kv_transfer_domain=f"nixl:{self._joint_run_id}",
            kv_transfer_namespace=f"{self._joint_run_id}:{version}",
        )

    def _joint_on_output(self, engine_id, output):
        if not self._joint_enabled:
            return
        state = self._joint_by_engine.get(engine_id)
        if state is None or not output.outputs:
            return
        result = output.outputs[0]
        state["tokens"] = list(result.token_ids)
        state["finish_reason"] = result.finish_reason

    async def get_request_progress(self, request_id):
        state = self._joint_active.get(request_id)
        if state is None:
            return {"found": False}
        return dict(
            found=True,
            generated_tokens=len(state["tokens"]),
            prefix_ids=state["prompt"] + state["tokens"],
            generation_id=state["engine_id"],
            weight_version=state["version"],
        )

    async def generate(
        self,
        prompt_ids,
        sampling_params,
        request_id,
        *generation_args,
        expected_source_version=None,
        migration_ticket=None,
        profile_context=None,
        **kwargs,
    ):
        if not self._joint_enabled:
            return await self._generate_impl(prompt_ids, sampling_params, request_id, *generation_args, **kwargs)
        if request_id in self._joint_active:
            raise RuntimeError("a prior generation still owns this logical request")
        version = self._joint_version()
        prompt = list(prompt_ids)
        if migration_ticket is not None:
            reason = self._joint_validate_ticket(migration_ticket, request_id, prompt)
            accepted = self._joint_accepted.pop(request_id, None)
            if reason is not None or accepted != migration_ticket:
                return TokenOutput(
                    token_ids=[],
                    log_probs=[],
                    stop_reason="aborted",
                    extra_fields={
                        "global_steps": self.global_steps,
                        "migration_rejected": True,
                        "migration_rejection_reason": reason or "acceptance no longer exists",
                    },
                )
        elif expected_source_version is not None and expected_source_version != version:
            raise RuntimeError("Joint trajectory continuation requires migration to the current weight version")
        context = dict(profile_context or {})
        context.setdefault("original_prompt_tokens", len(prompt))
        context.setdefault("generated_before", 0)
        context.update(weight_version=version, logical_request_id=str(request_id))
        engine_id = uuid.uuid4().hex
        context["engine_generation_id"] = engine_id
        state = dict(
            engine_id=engine_id,
            version=version,
            prompt=prompt,
            tokens=[],
            finish_reason=None,
            selected=False,
            aborted=False,
            done=asyncio.Event(),
        )
        self._joint_active[request_id] = state
        self._joint_by_engine[engine_id] = state
        params = dict(sampling_params)
        extra = dict(params.get("extra_args") or {})
        extra["v17_profile"] = context
        params["extra_args"] = extra
        if migration_ticket is not None and migration_ticket["backend"] == "remote_prefix":
            kwargs["kv_transfer_params"] = dict(migration_ticket["nixl_transfer"])
        self._joint_event(
            "request_admitted",
            engine_request_id=engine_id,
            context=context,
            prefix_tokens=len(prompt),
            migration_mode=(migration_ticket or {}).get("backend"),
        )
        try:
            result = await self._generate_impl(prompt, params, engine_id, *generation_args, **kwargs)
            result.extra_fields["global_steps"] = None if version == "initial" else version
            result.extra_fields["reached_eos"] = state["finish_reason"] == "stop"
            if (state["selected"] or state["aborted"]) and not result.extra_fields["reached_eos"]:
                result.stop_reason = "aborted"
            transfer = result.extra_fields.get("kv_transfer_params")
            if state["selected"] and transfer and not result.extra_fields["reached_eos"]:
                prefix = prompt + list(result.token_ids)
                self._joint_exports[request_id] = dict(
                    prefix_digest=prefix_digest(prefix),
                    prefix_tokens=len(prefix),
                    source_version=version,
                    transfer=dict(transfer),
                    expires_at=time.time() + self._joint_lease_seconds - 10,
                )
                result.extra_fields["migration_checkpoint"] = True
            if migration_ticket is not None and migration_ticket["backend"] == "remote_prefix":
                # Actual external tokens are accounted by the native scheduler
                # profiler; an accepted ticket alone is never called a cache hit.
                result.extra_fields["migration_transport"] = "nixl"
            self._joint_event(
                "request_completed",
                engine_request_id=engine_id,
                generated_tokens=len(result.token_ids),
                context=context,
                stop_reason=result.stop_reason,
                cached_tokens=result.extra_fields.get("num_cached_tokens"),
            )
            return result
        except BaseException as error:
            # Cancelling the Ray waiter does not cancel native GPU generation.
            # Keep ownership until vLLM acknowledges the abort.
            abort = asyncio.create_task(self.engine.abort(engine_id))
            while not abort.done():
                try:
                    await asyncio.shield(abort)
                except asyncio.CancelledError:
                    continue
            abort.result()
            self._joint_event(
                "request_failed", engine_request_id=engine_id, error=type(error).__name__, context=context
            )
            raise
        finally:
            self._joint_by_engine.pop(engine_id, None)
            self._joint_active.pop(request_id, None)
            state["done"].set()

    async def abort_request(
        self, request_id, reset_prefix_cache=True, *, expected_generation_id=None, preserve_kv=False, timeout_s=120
    ):
        if not self._joint_enabled:
            return await self._abort_request_impl(request_id, reset_prefix_cache=reset_prefix_cache)
        state = self._joint_active.get(request_id)
        if state is None:
            return {"aborted": False, "reason": "request not found on this replica"}
        if expected_generation_id is not None and state["engine_id"] != expected_generation_id:
            return {"aborted": False, "reason": "generation changed"}
        if expected_generation_id is not None:
            if self._submission_paused:
                return {"aborted": False, "reason": "replica update superseded selected checkpoint"}
            result = await self._joint_native(
                "v17_checkpoint_request", state["engine_id"], state["version"], preserve_kv
            )
            if not result.get("aborted"):
                return result
            state["selected"] = True
        else:
            state["aborted"] = True
            await self.engine.abort(state["engine_id"])
        await asyncio.wait_for(state["done"].wait(), timeout_s)
        return {"aborted": True, "preserve_kv": preserve_kv, "request_id": request_id}

    async def prepare_trajectory_migration(
        self, trajectory_state, target_metadata, *, backend="remote_prefix", timeout_s=30.0
    ):
        if backend != "remote_prefix":
            raise ValueError(f"vLLM trajectory export does not support backend {backend!r}")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("trajectory export timeout must be finite and positive")
        request_id = trajectory_state["request_id"]
        prefix = list(trajectory_state["prompt_ids"]) + list(trajectory_state["generated_token_ids"])
        export = self._joint_exports.get(request_id)
        source = await asyncio.wait_for(self.get_trajectory_migration_capabilities(), timeout=timeout_s)
        if export is None or export["expires_at"] <= time.time():
            raise RuntimeError("selected trajectory has no live NIXL export")
        if export["prefix_digest"] != prefix_digest(prefix) or export["prefix_tokens"] != len(prefix):
            raise ValueError("exported trajectory prefix differs from continuation")
        for key in (
            "model_id",
            "weight_version",
            "kv_transfer_domain",
            "kv_transfer_namespace",
            "rollout_config_fingerprint",
        ):
            if source[key] != target_metadata.get(key):
                raise ValueError(f"NIXL transfer requires matching {key}")
        if source["weight_version"] != export["source_version"] or target_metadata["replica_rank"] == self.replica_rank:
            raise ValueError("NIXL trajectory transfer requires distinct replicas at the exported weight version")
        transfer = {**export["transfer"], "v17_trajectory_migration": True}
        return dict(
            backend="remote_prefix",
            mode="remote_prefix",
            request_id=request_id,
            prefix_tokens=len(prefix),
            prefix_digest=export["prefix_digest"],
            model_id=source["model_id"],
            weight_version=source["weight_version"],
            kv_transfer_domain=source["kv_transfer_domain"],
            kv_transfer_namespace=source["kv_transfer_namespace"],
            source_replica_rank=self.replica_rank,
            target_replica_rank=target_metadata["replica_rank"],
            nixl_transfer=transfer,
            expires_at=export["expires_at"],
            explicit_export=True,
        )

    def _joint_validate_ticket(self, ticket, request_id, prefix):
        if self._submission_paused:
            return "target admission is closed"
        if (
            ticket.get("request_id") != request_id
            or ticket.get("prefix_digest") != prefix_digest(prefix)
            or ticket.get("prefix_tokens") != len(prefix)
        ):
            return "trajectory prefix identity differs"
        if ticket.get("model_id") != self.model_config.local_path:
            return "model identity differs"
        if ticket.get("backend") == "recompute":
            target = ticket.get("target_version")
            if not self.config.trajectory_migration.allow_cross_version_recompute:
                return "cross-version recompute disabled"
        elif ticket.get("backend") == "remote_prefix":
            target = ticket.get("weight_version")
            if (
                ticket.get("target_replica_rank") != self.replica_rank
                or ticket.get("source_replica_rank") == self.replica_rank
            ):
                return "transfer belongs to another replica"
            if ticket.get("expires_at", 0) <= time.time():
                return "KV transfer lease expired"
            if ticket.get("kv_transfer_domain") != f"nixl:{self._joint_run_id}":
                return "KV transfer domain differs"
            if ticket.get("kv_transfer_namespace") != f"{self._joint_run_id}:{self._joint_version()}":
                return "KV transfer namespace differs"
            if not ticket.get("explicit_export") or not ticket.get("nixl_transfer", {}).get("remote_block_ids"):
                return "missing explicit KV transfer lease"
        else:
            return "unsupported migration backend"
        if target != self._joint_version():
            return "target weight version changed"
        return None

    async def accept_trajectory_migration(self, ticket, trajectory_state):
        prefix = list(trajectory_state["prompt_ids"]) + list(trajectory_state["generated_token_ids"])
        reason = self._joint_validate_ticket(ticket, trajectory_state["request_id"], prefix)
        if reason:
            return {"accepted": False, "reason": reason}
        if len(self._joint_accepted) >= self.config.max_num_seqs * 2:
            return {"accepted": False, "reason": "migration acceptance queue is full"}
        self._joint_accepted[ticket["request_id"]] = dict(ticket)
        return {"accepted": True}

    async def discard_trajectory_migration(self, request_id, prefix_digest):
        ticket = self._joint_accepted.get(request_id)
        if ticket is None or ticket["prefix_digest"] != prefix_digest:
            return {"discarded": False}
        self._joint_accepted.pop(request_id)
        return {"discarded": True}

    async def _joint_before_pause(self):
        if not self._joint_enabled:
            return
        self._submission_paused = self._rejecting = True
        self._resume_event.clear()
        self._joint_accepted.clear()
        deadline = time.monotonic() + self._joint_lease_seconds + 30
        while (await self._joint_native("v17_scheduler_state"))["pending_kv_exports"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("source KV leases did not settle; refusing weight installation")
            await asyncio.sleep(0.02)
        self._joint_exports.clear()
        for state in self._joint_active.values():
            state["aborted"] = True

    async def seal_reprefill_profile(self):
        result = await self.engine.collective_rpc("v17_seal_profile")
        self._joint_event("profile_sealed", native=result)
        self._joint_profile_file.flush()
        return result


def ray_node_ip():
    import ray

    return ray.util.get_node_ip_address()
