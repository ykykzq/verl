import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.learning_acceptance import ParameterUpdateProbe, learning_batch_metrics


def test_global_reward_mean_does_not_prove_group_variance():
    rewards = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    with pytest.raises(ValueError, match="group reward variance"):
        learning_batch_metrics(
            rewards, torch.ones_like(rewards), torch.ones_like(rewards), ["a", "a", "b", "b"], strict=True
        )


def test_masked_nonzero_advantage_does_not_pass():
    with pytest.raises(ValueError, match="effective advantage"):
        learning_batch_metrics(
            torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            torch.tensor([[0.0, 1.0], [0.0, -1.0]]),
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            ["a", "a"],
            strict=True,
        )


def test_real_math_rewards_and_grpo_gradient_change_parameter():
    import numpy as np

    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.utils.reward_score.math_dapo import compute_score

    rewards = torch.tensor(
        [[compute_score(r"\boxed{4}", "4")["score"]], [compute_score(r"\boxed{5}", "4")["score"]]], dtype=torch.float32
    )
    mask = torch.ones_like(rewards)
    advantages, _ = compute_grpo_outcome_advantage(rewards, mask, np.array(["a", "a"]))
    metrics = learning_batch_metrics(rewards, advantages, mask, ["a", "a"], strict=True)
    assert metrics["learning/groups_with_reward_variance"] == 1
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    loss = -(parameter.log_softmax(0) * advantages.flatten()).mean()
    loss.backward()
    probe = ParameterUpdateProbe([parameter])
    optimizer.step()
    assert probe.finish(parameter.grad.norm().item())["parameter_delta_max"] > 0


def test_unchanged_parameter_fails_even_with_nonzero_gradient():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.sum().backward()
    probe = ParameterUpdateProbe([parameter])
    with pytest.raises(ValueError, match="parameters did not change"):
        probe.finish(1.0)


def test_weight_decay_alone_does_not_pass():
    parameter = torch.nn.Parameter(torch.ones(2))
    probe = ParameterUpdateProbe([parameter])
    with torch.no_grad():
        parameter.mul_(0.9)
    with pytest.raises(ValueError, match="gradient"):
        probe.finish(0.0)


@pytest.mark.parametrize(
    "uids,rewards,message,counts",
    [
        (["a", "b", "c", "d"], [0.0, 1.0, 0.0, 1.0], "no multi-sample UID groups", (4, 4, 0)),
        (["a", "a", "b", "b"], [0.0, 0.0, 1.0, 1.0], "group reward variance", (2, 0, 2)),
    ],
)
def test_failed_acceptance_retains_group_and_token_evidence(uids, rewards, message, counts):
    rewards = torch.tensor(rewards).unsqueeze(-1)
    with pytest.raises(ValueError, match=message) as failure:
        learning_batch_metrics(rewards, torch.zeros_like(rewards), torch.ones_like(rewards), uids, strict=True)

    error = failure.value
    assert error.metrics["learning/uid_groups"] == counts[0]
    assert error.metrics["learning/singleton_uid_groups"] == counts[1]
    assert error.metrics["learning/multisample_uid_groups"] == counts[2]
    assert error.metrics["learning/effective_tokens"] == 4
    assert error.metrics["learning/nonzero_advantage_tokens"] == 0
    assert error.diagnostics["groups"][0]["uid"] == "a"
    assert error.diagnostics["groups"][0]["rewards"] == ([0.0] if counts[1] else [0.0, 0.0])
    assert error.diagnostics["samples"][0]["nonzero_advantage_tokens"] == 0


def test_varied_rewards_report_group_sizes_without_weakening_strict_checks():
    rewards = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0]])
    metrics = learning_batch_metrics(
        rewards, torch.ones_like(rewards), torch.ones_like(rewards), ["a", "a", "a", "a", "b"], strict=True
    )
    assert metrics["learning/uid_groups"] == 2
    assert metrics["learning/singleton_uid_groups"] == 1
    assert metrics["learning/multisample_uid_groups"] == 1
    assert metrics["learning/uid_group_size_min"] == 1
    assert metrics["learning/uid_group_size_max"] == 4
    assert metrics["learning/groups_with_reward_variance"] == 1


def test_failed_acceptance_evidence_is_bounded():
    rewards = torch.zeros(2048, 1)
    uids = [f"group-{i // 32}" for i in range(2048)]
    with pytest.raises(ValueError) as failure:
        learning_batch_metrics(rewards, rewards, torch.ones_like(rewards), uids, strict=True)
    evidence = failure.value.diagnostics
    assert len(evidence["samples"]) <= 8
    assert len(evidence["groups"]) <= 32
    assert sum(len(group["rewards"]) for group in evidence["groups"]) <= 512
    assert evidence["samples_omitted"] == 2040
    assert evidence["groups_omitted"] == 32
    assert evidence["groups"][0]["sample_count"] == 32
    assert evidence["groups"][0]["rewards_omitted"] == 16


@pytest.mark.parametrize("logger_fails", [False, True])
def test_trainer_writes_failure_evidence_before_actor_update(tmp_path, logger_fails, capsys):
    from omegaconf import OmegaConf

    from verl import DataProto
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    trainer = SeparateRayPPOTrainer.__new__(SeparateRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"critic_warmup": 0, "learning_acceptance": True, "default_local_dir": str(tmp_path)},
            "actor_rollout_ref": {
                "actor": {"optim": {"verify_parameter_updates": True, "weight_decay": 0}},
                "rollout": {"verify_weight_updates": True, "response_length": 4},
            },
        }
    )
    trainer.global_steps = 3
    trainer.metrics = {}
    trainer.timing_raw = {}
    trainer.reward_extra_infos_dict = {"acc": [0.0, 0.0], "prediction": ["x" * 10000, "wrong"]}
    trainer.tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda ids, **kwargs: " ".join(map(str, ids)))
    logged_metrics = []

    def record_metrics(data, step):
        logged_metrics.append((data, step))
        if logger_fails:
            raise RuntimeError("tracking backend unavailable")

    trainer.logger = SimpleNamespace(log=record_metrics)

    def unexpected_update(batch):
        pytest.fail("strict acceptance must prevent the actor update")

    trainer._update_actor = unexpected_update
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[10, 11, 12, 13], [14, 2, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
            "rollout_is_weights": torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
            "token_level_scores": torch.zeros(2, 4),
            "advantages": torch.zeros(2, 4),
        },
        non_tensors={
            "uid": np.array(["a", "a"], dtype=object),
            "stop_reason": np.array(["length", "completed"], dtype=object),
            "reward_model": np.array([{"ground_truth": "4"}, {"ground_truth": "4"}], dtype=object),
        },
    )
    with pytest.raises(ValueError, match="group reward variance") as failure:
        trainer._fit_update_actor(batch)

    evidence_files = list(tmp_path.rglob("*.json"))
    assert len(evidence_files) == 1
    evidence = json.loads(evidence_files[0].read_text())
    assert evidence["step"] == 3
    assert evidence["metrics"]["learning/effective_tokens"] == 2
    assert evidence["groups"][0]["sample_count"] == 2
    assert evidence["samples"][0]["response_length"] == 4
    assert evidence["samples"][0]["possible_length_truncation"] is True
    assert evidence["samples"][0]["ended_with_eos"] is False
    assert evidence["samples"][0]["stop_reason"] == "length"
    assert evidence["samples"][0]["reward_evidence"]["acc"] == 0.0
    assert len(evidence["samples"][0]["reward_evidence"]["prediction"]) <= 1024
    assert evidence["samples"][0]["ground_truth"] == "4"
    assert evidence["samples"][1]["response_length"] == 2
    assert evidence["samples"][1]["ended_with_eos"] is True
    assert evidence["samples"][1]["possible_length_truncation"] is False
    assert failure.value.diagnostics["samples"][0]["response_token_ids"] == [10, 11, 12, 13]
    assert trainer.metrics["learning/uid_groups"] == 1
    assert logged_metrics[0][0]["learning/uid_groups"] == 1
    assert logged_metrics[0][1] == 3
    if logger_fails:
        assert "tracking backend unavailable" in capsys.readouterr().out


def test_nonfinite_failure_evidence_remains_json_readable_and_survives_pickling():
    rewards = torch.tensor([[float("nan")], [1.0]])
    with pytest.raises(ValueError, match="nonfinite") as failure:
        learning_batch_metrics(rewards, torch.ones_like(rewards), torch.ones_like(rewards), ["a", "a"], strict=True)

    error = pickle.loads(pickle.dumps(failure.value))
    assert error.metrics["learning/finite"] == 0
    evidence = json.loads(json.dumps(error.diagnostics, allow_nan=False))
    assert evidence["groups"][0]["rewards"] == ["nan", 1.0]
    assert evidence["samples"][0]["reward"] == "nan"


def test_long_response_evidence_preserves_bounded_prefix_and_answer_tail():
    from verl import DataProto
    from verl.trainer.ppo.learning_acceptance import add_learning_batch_evidence

    rewards = torch.zeros(2, 300)
    with pytest.raises(ValueError) as failure:
        learning_batch_metrics(rewards, rewards, torch.ones_like(rewards), ["a", "a"], strict=True)
    batch = DataProto.from_dict(
        tensors={"responses": torch.arange(300).repeat(2, 1), "response_mask": torch.ones(2, 300)},
    )
    tokenizer = SimpleNamespace(eos_token_id=299, decode=lambda ids, **kwargs: " ".join(map(str, ids)))
    add_learning_batch_evidence(failure.value.diagnostics, batch, tokenizer, {}, 300)
    sample = failure.value.diagnostics["samples"][0]
    assert len(sample["response_token_ids"]) == 128
    assert sample["response_token_ids"][:3] == [0, 1, 2]
    assert sample["response_token_ids"][-3:] == [297, 298, 299]
    assert sample["response_tokens_omitted"] == 172
    assert len(sample["response_text"]) <= 1024
    assert sample["response_text"].endswith("297 298 299")
    assert sample["ended_with_eos"] is True
    assert sample["possible_length_truncation"] is False
    assert sample["stop_reason"] is None
