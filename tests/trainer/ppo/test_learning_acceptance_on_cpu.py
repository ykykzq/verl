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
