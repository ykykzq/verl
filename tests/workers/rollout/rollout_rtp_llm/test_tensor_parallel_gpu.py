# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in four-GPU acceptance using a local, small, dense Qwen checkpoint."""

import asyncio
import hashlib
import os
import time

import pytest
import ray
import torch

from verl.checkpoint_engine.weight_snapshot import export_weight_snapshot
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.config.rollout import CheckpointEngineConfig
from verl.workers.rollout.rtp_llm_rollout.rtp_llm_async_server import RTPLLMReplica


def rank_fingerprint(server):
    """Inspect real shard bytes, not only the adapter's version marker."""
    weights = server.weight_manager._weights.weights[0]
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        if tensor is not None:
            digest.update(name.encode())
            sample = tensor.detach().reshape(-1)[:4096].contiguous().cpu()
            digest.update(sample.view(torch.uint8).numpy().tobytes())
    pc = server._cfg.parallelism_config
    return {"rank": pc.tp_rank, "size": pc.tp_size, "device": torch.cuda.current_device(), "digest": digest.hexdigest()}


@pytest.mark.skipif(
    os.environ.get("VERL_RUN_RTP_TP_TEST") != "1",
    reason="requires explicit opt-in, four CUDA GPUs, RTP-LLM and a local dense Qwen checkpoint",
)
@pytest.mark.asyncio
async def test_tp4_generation_weight_update_and_rank_failure():
    from transformers import AutoModelForCausalLM

    path = os.environ["VERL_RTP_TP_TEST_MODEL"]
    connected_here = not ray.is_initialized()
    replica = None
    try:
        if connected_here:
            ray.init(address=os.environ.get("RAY_ADDRESS"), ignore_reinit_error=False)
        assert any(node["Alive"] and node["Resources"].get("GPU", 0) >= 4 for node in ray.nodes())
        model = HFModelConfig(path=path, load_tokenizer=True)
        config = RolloutConfig(
            name="rtp_llm",
            tensor_model_parallel_size=4,
            dtype="bfloat16",
            prompt_length=128,
            response_length=16,
            max_model_len=256,
            max_num_seqs=2,
            gpu_memory_utilization=0.75,
            checkpoint_engine=CheckpointEngineConfig(backend="nccl", update_weights_bucket_megabytes=64),
        )
        replica = RTPLLMReplica(0, config, model, gpus_per_node=4)
        await asyncio.wait_for(replica.init_standalone(), 900)
        prompt = model.tokenizer.encode("The capital of France is", add_special_tokens=True)
        sampling = {"max_tokens": 8, "temperature": 0, "logprobs": True}
        output = await asyncio.wait_for(replica.server_handle.generate.remote(prompt, sampling, "before"), 60)
        assert output.token_ids and len(output.log_probs) == len(output.token_ids)
        before = await asyncio.gather(*[s.__ray_call__.remote(rank_fingerprint) for s in replica.servers])
        assert [(r["rank"], r["size"], r["device"]) for r in before] == [(i, 4, i) for i in range(4)]

        # Small models only: this test deliberately materializes a CPU reference.
        reference = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cpu")
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.mul_(1.01)
        snapshot = export_weight_snapshot(reference.state_dict().items(), 7, 64 << 20, is_master=True)
        del reference
        await replica.abort_all_requests(reject_request=True)
        await replica.release_kv_cache()
        await asyncio.wait_for(
            asyncio.gather(*[worker.update_weights_from_snapshot.remote(snapshot) for worker in replica.workers]), 180
        )
        await replica.resume_generation()
        statuses = await asyncio.gather(*[s.get_tp_weight_status.remote() for s in replica.servers])
        assert all(s["weight_version"] == 7 and not s["updating"] and s["error"] is None for s in statuses)
        after = await asyncio.gather(*[s.__ray_call__.remote(rank_fingerprint) for s in replica.servers])
        assert all(old["digest"] != new["digest"] for old, new in zip(before, after, strict=True))
        output = await asyncio.wait_for(replica.server_handle.generate.remote(prompt, sampling, "after"), 60)
        assert output.token_ids and output.extra_fields["global_steps"] == 7

        ray.kill(replica.servers[2], no_restart=True)
        deadline = time.monotonic() + 45
        while True:
            try:
                await replica.server_handle.get_tp_health.remote()
            except ray.exceptions.RayError:
                break
            assert time.monotonic() < deadline, "leader did not detect its failed TP rank"
            await asyncio.sleep(0.2)
        with pytest.raises(ray.exceptions.RayError):
            await asyncio.wait_for(replica.server_handle.generate.remote(prompt, sampling, "failed"), 5)
    finally:
        if replica is not None:
            for actor in [*replica.servers, *replica.workers]:
                ray.kill(actor, no_restart=True)
            if replica.resource_pool is not None:
                for group in replica.resource_pool.pgs or []:
                    ray.util.remove_placement_group(group)
        if connected_here and ray.is_initialized():
            ray.shutdown()
