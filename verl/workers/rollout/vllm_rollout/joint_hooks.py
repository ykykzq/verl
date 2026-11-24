"""Explicitly enabled hooks for the pinned vLLM 0.32.0 Joint experiment.

No background KV writes: only a selected request is marked for NIXL export.
The original scheduler still chooses all batches, budgets and preemptions.
"""

import os


def checkpoint_request(core, request_id, weight_version, preserve_kv):
    request = core.scheduler.requests.get(request_id)
    if request is None:
        matches = [
            r
            for r in core.scheduler.requests.values()
            if (r.sampling_params.extra_args or {}).get("v17_profile", {}).get("engine_generation_id") == request_id
        ]
        if len(matches) > 1:
            raise RuntimeError("ambiguous backend generation identity")
        request = matches[0] if matches else None
    if request is None or request.is_finished():
        return {"aborted": False, "reason": "request already finished"}
    extra = request.sampling_params.extra_args or {}
    if extra.get("v17_profile", {}).get("weight_version") != weight_version:
        return {"aborted": False, "reason": "weight version changed"}
    if request.num_output_tokens < 1:
        return {"aborted": False, "reason": "request has not reached decode"}
    prior = extra.get("v17_checkpoint")
    if prior is not None:
        return {"aborted": True, "preserve_kv": prior["preserve_kv"]}
    if request.num_output_tokens + 1 >= request.max_tokens:
        return {"aborted": False, "reason": "request is already at its final token budget"}
    if preserve_kv:
        connector = core.scheduler.get_kv_connector()
        scheduler = getattr(connector, "connector_scheduler", None)
        if type(connector).__name__ != "JointNixlConnector" or scheduler is None or scheduler.use_host_buffer:
            raise RuntimeError("selected migration requires the validated direct-GPU JointNixlConnector")
        request.kv_transfer_params = {"do_remote_decode": True, "do_remote_prefill": False}
        scheduler._reqs_in_batch.add(request.request_id)
    # Finish through vLLM's normal output/connector path at the next model step.
    # An ordinary abort would free the blocks without returning a transfer lease.
    request.max_tokens = min(request.max_tokens, request.num_output_tokens + 1)
    request.sampling_params.max_tokens = request.max_tokens
    extra["v17_checkpoint"] = {"preserve_kv": bool(preserve_kv)}
    request.sampling_params.extra_args = extra
    return {"aborted": True, "preserve_kv": bool(preserve_kv)}


def scheduler_state(core):
    scheduler = core.scheduler
    pending = []
    for rid, request in scheduler.requests.items():
        extra = request.sampling_params.extra_args or {}
        if extra.get("v17_checkpoint", {}).get("preserve_kv"):
            pending.append(rid)
    cache = scheduler.vllm_config.cache_config
    return {
        "pending_kv_exports": pending,
        "kv_capacity_tokens": int(cache.num_gpu_blocks) * scheduler.block_size,
        "kv_block_size_tokens": scheduler.block_size,
        "max_num_seqs": scheduler.max_num_running_reqs,
        "max_model_len": scheduler.max_model_len,
    }


def kv_cache_usage(core):
    return float(core.scheduler.kv_cache_manager.usage)


def register():
    if os.environ.get("VERL_V17_JOINT_ENABLE") != "1":
        return
    from verl.workers.rollout.vllm_rollout.native_libraries import load_nixl_dependencies

    load_nixl_dependencies()
    import vllm

    if vllm.__version__ != "0.32.0":
        raise RuntimeError("V17 Joint hooks were validated only against vLLM 0.32.0")
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import RequestStatus
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(EngineCore, "_v17_joint_registered", False):
        return
    EngineCore.v17_checkpoint_request = checkpoint_request
    EngineCore.v17_scheduler_state = scheduler_state
    EngineCore.verl_kv_cache_usage = kv_cache_usage

    original_finish = Scheduler._connector_finished

    def connector_finished(self, request):
        extra = request.sampling_params.extra_args or {}
        if extra.get("v17_checkpoint", {}).get("preserve_kv") and request.status == RequestStatus.FINISHED_STOPPED:
            # EOS won the selected-checkpoint race: no continuation will consume
            # this lease, so finish normally without pinning KV until a timeout.
            connector_scheduler = self.get_kv_connector().connector_scheduler
            connector_scheduler._reqs_in_batch.discard(request.request_id)
            connector_scheduler._reqs_not_processed.add(request.request_id)
            request.kv_transfer_params = None
            extra.pop("v17_checkpoint", None)
        return original_finish(self, request)

    Scheduler._connector_finished = connector_finished
    original_execute = Worker.execute_model

    def execute_model(self, scheduler_output):
        directory = os.environ.get("VERL_V17_REPREFILL_PROFILE_DIR")
        if not directory:
            raise RuntimeError("V17 Joint requires a native re-prefill profile directory")
        if not hasattr(self, "_v17_forward_profile"):
            from verl.workers.rollout.vllm_rollout.reprefill_profile import CUDAForwardProfile

            self._v17_forward_profile = CUDAForwardProfile(directory, int(os.environ["VERL_REPLICA_RANK"]), self.device)
        meter = self._v17_forward_profile
        ticket = meter.start(scheduler_output)
        try:
            result = original_execute(self, scheduler_output)
        except BaseException:
            meter.finish(ticket, failed=True)
            raise
        meter.finish(ticket)
        return result

    def seal_profile(self):
        meter = getattr(self, "_v17_forward_profile", None)
        return meter.seal() if meter is not None else {"sealed": True, "forwards": 0}

    Worker.execute_model = execute_model
    Worker.v17_seal_profile = seal_profile
    EngineCore._v17_joint_registered = True
