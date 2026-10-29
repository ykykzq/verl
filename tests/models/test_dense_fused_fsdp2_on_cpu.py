# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Real two-rank FSDP2 regression for the dense fused PPO output head.

CPU coverage requires a PyTorch build supporting FSDP2 on CPU. For the
accelerator stack, run all cases with:

    torchrun --standalone --nproc-per-node=2 tests/models/test_dense_fused_fsdp2_on_cpu.py \
        --device cuda --dtype bfloat16 --backend torch

Repeat with --backend triton or liger when those kernels are installed.
"""

import argparse
import copy
import os
from datetime import timedelta
from types import MethodType

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2


def _standard_outputs(model, input_ids, labels, temperature):
    logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits / temperature
    log_probs = logits.float().log_softmax(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1), entropy


def _full_tensor(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def _check_case(device, dtype, backend, tie_word_embeddings, reshard_after_forward):
    torch.manual_seed(123)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh(device.type, (world_size,))
    hidden_size = 128 if backend == "triton" else 16
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tie_word_embeddings,
        attention_dropout=0.0,
    )
    reference = Qwen2ForCausalLM(config).to(device=device, dtype=dtype)
    reference.config._attn_implementation = "eager"
    model = copy.deepcopy(reference)
    forward = forward_with_triton_backend if backend == "triton" else forward_with_torch_backend
    model.forward = MethodType(forward, model)
    model._verl_fused_kernels_backend = backend
    apply_fsdp2(
        model,
        {
            "mesh": mesh,
            "mp_policy": MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
            "reshard_after_forward": reshard_after_forward,
        },
        {},
    )
    assert isinstance(model.lm_head.weight, DTensor)
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == tie_word_embeddings
    names_before = set(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    initial_head = reference.lm_head.weight.detach().clone()
    tolerance = {"atol": 3e-6, "rtol": 3e-5} if dtype == torch.float32 else {"atol": 3e-3, "rtol": 3e-2}
    temperature = 0.7

    for step in range(2):
        # Distinct rank inputs require real gradient averaging across the mesh.
        inputs = (torch.arange(world_size * 12, device=device).reshape(world_size * 2, 6) + step) % 32
        labels = (inputs + 3) % 32  # Explicit globally prepared labels, distinct from a local roll.
        rank_slice = slice(rank * 2, (rank + 1) * 2)
        kwargs = {
            "input_ids": inputs[rank_slice],
            "shift_labels": labels[rank_slice],
            "temperature": temperature,
            "return_dict": True,
            "use_cache": False,
        }
        with torch.no_grad():
            expected_log_probs, expected_entropy = _standard_outputs(reference, inputs, labels, temperature)
            # compute_log_prob can execute repeatedly before actor training.
            for _ in range(2):
                output = model(**kwargs)
                torch.testing.assert_close(
                    output.log_probs.flatten(), expected_log_probs[rank_slice].flatten(), **tolerance
                )
                torch.testing.assert_close(
                    output.entropy.float().flatten(), expected_entropy[rank_slice].flatten(), **tolerance
                )
                if not tie_word_embeddings and reshard_after_forward:
                    assert isinstance(model.lm_head.weight, DTensor)

        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        output = model(**kwargs)
        expected_log_probs, expected_entropy = _standard_outputs(reference, inputs, labels, temperature)
        (-output.log_probs.mean() - 0.1 * output.entropy.float().mean()).backward()
        (-expected_log_probs.mean() - 0.1 * expected_entropy.mean()).backward()

        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            torch.testing.assert_close(
                _full_tensor(parameter.grad), reference_parameters[name].grad, **tolerance, msg=name
            )
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(_full_tensor(parameter), reference_parameters[name], **tolerance, msg=name)

    assert not torch.equal(reference.lm_head.weight, initial_head), "optimizer must update the output head"
    assert not torch.equal(_full_tensor(model.lm_head.weight), initial_head), (
        "FSDP optimizer must update the output head"
    )
    assert set(model.state_dict()) == names_before, "fused head must preserve checkpoint parameter names"


def _cpu_worker(rank, rendezvous, tie_word_embeddings, reshard_after_forward):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        _check_case(torch.device("cpu"), torch.float32, "torch", tie_word_embeddings, reshard_after_forward)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(torch.cpu.Stream, "wait_event"),
    reason="CPU FSDP2 requires torch.cpu.Stream.wait_event, which older PyTorch builds do not provide",
)
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@pytest.mark.parametrize("reshard_after_forward", [True, False])
def test_dense_fused_fsdp2_matches_standard_forward(tmp_path, tie_word_embeddings, reshard_after_forward):
    mp.spawn(
        _cpu_worker,
        args=((tmp_path / "rendezvous").as_uri(), tie_word_embeddings, reshard_after_forward),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--backend", choices=("torch", "triton", "liger"), default="torch")
    args = parser.parse_args()
    if args.backend == "liger":
        from verl.utils.experimental.torch_functional import _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY

        assert _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY is not None, "Liger kernel is required for this backend test"
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    try:
        assert dist.get_world_size() >= 2, "run with at least two ranks"
        for tied in (False, True):
            for reshard in (True, False):
                _check_case(device, getattr(torch, args.dtype), args.backend, tied, reshard)
                if dist.get_rank() == 0:
                    print(f"passed: backend={args.backend}, tied={tied}, reshard={reshard}")
    finally:
        dist.destroy_process_group()
