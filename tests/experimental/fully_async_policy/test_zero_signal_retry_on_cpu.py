import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from verl.experimental.fully_async_policy.batch_acceptance import prepare_learning_batch
from verl.trainer.ppo.learning_acceptance import LearningAcceptanceError, learning_batch_metrics


def batch(rewards=(0.0, 0.0), advantages=(0.0, 0.0), uids=("a", "a")):
    return SimpleNamespace(
        batch={
            "token_level_scores": torch.tensor(rewards).reshape(-1, 1),
            "advantages": torch.tensor(advantages).reshape(-1, 1),
            "response_mask": torch.ones(len(rewards), 1),
            "responses": torch.ones(len(rewards), 1, dtype=torch.long),
            "prompts": torch.arange(len(rewards)).reshape(-1, 1),
        },
        non_tensor_batch={"uid": uids},
    )


class FakeTrainer:
    def __init__(self, tmp_path, batches, limit=2):
        self.config = SimpleNamespace(
            trainer={
                "learning_acceptance": True,
                "learning_acceptance_max_zero_batches": limit,
                "rollout_data_dir": str(tmp_path),
            },
            algorithm=SimpleNamespace(adv_estimator="grpo"),
            actor_rollout_ref=SimpleNamespace(rollout={"response_length": 8}),
        )
        self.tokenizer = SimpleNamespace(
            eos_token_id=2, decode=lambda p, **k: str(p.tolist() if hasattr(p, "tolist") else p)
        )
        self.use_critic = False
        self.global_steps = 10
        self.current_param_version = 9
        self.metrics = {}
        self.calls = []
        self.remaining = iter(batches)
        self.logged_errors = []

    async def _fit_generate(self, _):
        assert self.future_reward is None and self.reward_tensor is None and not self.reward_extra_infos_dict
        self.calls.append("sample")
        return next(self.remaining)

    def _prepare(self, value):
        self.calls.append("prepare")
        return value

    _fit_compute_reward = _prepare
    _fit_compute_log_prob = _prepare
    _fit_compute_ref_log_prob = _prepare
    _fit_compute_critic = _prepare
    _fit_compute_advantage = _prepare

    def _log_learning_acceptance_failure(self, _, error):
        self.logged_errors.append(error)


def test_skip_zero_signal_then_keep_exact_valid_batch_without_advancing_state(tmp_path):
    valid = batch((0.0, 1.0), (-1.0, 1.0))
    trainer = FakeTrainer(tmp_path, [batch(), batch((1.0, 1.0)), valid])
    assert asyncio.run(prepare_learning_batch(trainer)) is valid
    assert (trainer.global_steps, trainer.current_param_version) == (10, 9)
    assert trainer.calls.count("sample") == 3
    assert trainer.metrics["learning/rejected_zero_batches"] == 2
    assert trainer.metrics["learning/rejected_zero_tokens"] == 4
    assert trainer.metrics["learning/rejected_zero_seconds"] > 0
    evidence = [json.loads(p.read_text()) for p in sorted((tmp_path / "learning_acceptance").glob("*.json"))]
    assert [x["attempt"] for x in evidence] == [1, 2]
    assert all(x["will_retry"] and not x["optimizer_executed"] for x in evidence)
    assert all(len(x["consumed_prompts"]) == 2 for x in evidence)


def test_zero_retry_budget_preserves_strict_actor_gate(tmp_path):
    rejected = batch()
    trainer = FakeTrainer(tmp_path, [rejected], limit=0)
    assert asyncio.run(prepare_learning_batch(trainer)) is rejected
    with pytest.raises(LearningAcceptanceError):
        learning_batch_metrics(
            rejected.batch["token_level_scores"],
            rejected.batch["advantages"],
            rejected.batch["response_mask"],
            rejected.non_tensor_batch["uid"],
            strict=True,
        )
    assert not (tmp_path / "learning_acceptance").exists()


def test_budget_exhaustion_fails_before_optimizer_and_preserves_all_evidence(tmp_path):
    trainer = FakeTrainer(tmp_path, [batch(), batch(), batch()], limit=1)
    with pytest.raises(LearningAcceptanceError):
        asyncio.run(prepare_learning_batch(trainer))
    assert trainer.calls.count("sample") == 2
    assert (trainer.global_steps, trainer.current_param_version) == (10, 9)
    evidence = json.loads((tmp_path / "learning_acceptance/10.rejected-2.json").read_text())
    assert not evidence["will_retry"] and not evidence["optimizer_executed"]


@pytest.mark.parametrize("value", [batch((float("nan"), 0.0)), batch(uids=("a", "b")), batch((0.0, 1.0))])
def test_nonfinite_broken_groups_and_zero_effective_is_signal_are_not_retried(tmp_path, value):
    trainer = FakeTrainer(tmp_path, [value, batch((0.0, 1.0), (-1.0, 1.0))])
    with pytest.raises(LearningAcceptanceError):
        asyncio.run(prepare_learning_batch(trainer))
    assert trainer.calls.count("sample") == 1
    assert len(trainer.logged_errors) == 1


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_retry_budget_rejected_before_consuming_data(tmp_path, limit):
    trainer = FakeTrainer(tmp_path, [], limit=limit)
    with pytest.raises(ValueError):
        asyncio.run(prepare_learning_batch(trainer))
    assert trainer.calls == []


def test_critic_or_non_grpo_cannot_retry_after_mutation(tmp_path):
    trainer = FakeTrainer(tmp_path, [])
    trainer.use_critic = True
    with pytest.raises(ValueError, match="without a critic"):
        asyncio.run(prepare_learning_batch(trainer))
    trainer.use_critic = False
    trainer.config.algorithm.adv_estimator = "gae"
    with pytest.raises(ValueError, match="without a critic"):
        asyncio.run(prepare_learning_batch(trainer))
    assert trainer.calls == []


def test_audit_collision_is_fatal_and_does_not_consume_next_batch(tmp_path):
    root = tmp_path / "learning_acceptance"
    root.mkdir()
    (root / "10.rejected-1.json").write_text("existing evidence")
    trainer = FakeTrainer(tmp_path, [batch(), batch((0.0, 1.0), (-1.0, 1.0))])
    with pytest.raises(FileExistsError):
        asyncio.run(prepare_learning_batch(trainer))
    assert trainer.calls.count("sample") == 1
    assert (root / "10.rejected-1.json").read_text() == "existing evidence"
