"""Keep each repeated prompt together through rollout serialization and assembly."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    assemble_batch_from_rollout_samples,
    prepare_single_generation_data,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.learning_acceptance import learning_batch_metrics


@pytest.mark.asyncio
async def test_four_responses_keep_their_prompt_uid_after_generation_and_assembly():
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 4, "multi_turn": {"enable": False}}}})
    rollouter_class = FullyAsyncRollouter.__ray_metadata__.modified_class
    rollouter = rollouter_class.__new__(rollouter_class)
    queued = []

    async def generate(batch):
        assert len(batch) == 4
        # Streaming reward workers return a fresh DataProto without the input UID.
        return DataProto.from_dict(
            tensors={
                "responses": torch.tensor([[10, 2]] * 4),
                "response_mask": torch.ones(4, 2, dtype=torch.long),
                "attention_mask": torch.ones(4, 3, dtype=torch.long),
                "token_level_scores": torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]),
            },
            non_tensors={"min_global_steps": np.zeros(4), "max_global_steps": np.zeros(4)},
            meta_info={"metrics": [{"generate_sequences": 0.1, "tool_calls": 0.0}] * 4},
        )

    async def put_sample(sample):
        queued.append(ray.cloudpickle.loads(sample))
        return True

    async def statistics():
        return {}

    rollouter.async_rollout_manager = SimpleNamespace(generate_sequences_single=generate)
    rollouter.message_queue_client = SimpleNamespace(put_sample=put_sample)
    rollouter.get_statistics = statistics
    rollouter.total_generated_samples = 0
    rollouter._step_generated_samples = 0
    rollouter.processed_sample_count = 0
    for index in range(2):
        prompts = prepare_single_generation_data({"input_ids": torch.tensor([[index]])}, config)
        await rollouter._process_single_sample_streaming(RolloutSample(prompts, f"sample_{index}", 0, {}))

    batch = assemble_batch_from_rollout_samples(queued, None, config)
    assert Counter(batch.non_tensor_batch["uid"]) == {"uid_sample_0": 4, "uid_sample_1": 4}
    advantages, _ = compute_grpo_outcome_advantage(
        batch.batch["token_level_scores"], batch.batch["response_mask"], batch.non_tensor_batch["uid"]
    )
    metrics = learning_batch_metrics(
        batch.batch["token_level_scores"],
        advantages,
        batch.batch["response_mask"],
        batch.non_tensor_batch["uid"],
        strict=True,
    )
    assert metrics["learning/groups_with_reward_variance"] == 2
    assert metrics["learning/nonzero_advantage_tokens"] == 16
