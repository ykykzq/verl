"""Bounded, audited resampling before any GRPO optimizer mutation."""

import hashlib
import json
import time
from pathlib import Path

from verl.trainer.ppo.learning_acceptance import (
    LearningAcceptanceError,
    add_learning_batch_evidence,
    learning_batch_metrics,
)


def is_zero_signal_batch(error):
    """Do not disguise nonfinite values, broken groups or masked IS failures."""
    metrics = error.metrics
    return (
        "no nonzero within-group reward variance" in str(error)
        and metrics.get("learning/finite") == 1
        and metrics.get("learning/multisample_uid_groups", 0) > 0
        and metrics.get("learning/singleton_uid_groups") == 0
        and metrics.get("learning/effective_tokens", 0) > 0
        and metrics.get("learning/groups_with_reward_variance") == 0
        and metrics.get("learning/nonzero_advantage_tokens") == 0
    )


def write_rejection(trainer, batch, error, attempt, will_retry, seconds):
    """Keep every rejected prompt hash, without overwriting earlier attempts."""
    evidence = dict(error.diagnostics)
    evidence.update(
        step=trainer.global_steps,
        weight_version=trainer.current_param_version,
        attempt=attempt,
        will_retry=will_retry,
        optimizer_executed=False,
        preparation_seconds=seconds,
        error=str(error),
    )
    add_learning_batch_evidence(
        evidence,
        batch,
        trainer.tokenizer,
        trainer.reward_extra_infos_dict,
        trainer.config.actor_rollout_ref.rollout.get("response_length"),
    )
    evidence["consumed_prompts"] = [
        {
            "uid": str(uid),
            "prompt_sha256": hashlib.sha256(
                trainer.tokenizer.decode(prompt, skip_special_tokens=True).encode()
            ).hexdigest(),
        }
        for uid, prompt in zip(batch.non_tensor_batch["uid"], batch.batch["prompts"], strict=True)
    ]
    root = trainer.config.trainer.get("rollout_data_dir") or trainer.config.trainer.default_local_dir
    path = Path(root) / "learning_acceptance" / f"{trainer.global_steps}.rejected-{attempt}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # An audit failure aborts the run instead of silently losing consumed data.
    with path.open("x", encoding="utf-8") as output:
        json.dump(evidence, output, ensure_ascii=False, allow_nan=False)
    print(f"Zero-signal batch evidence written to {path}; will_retry={will_retry}", flush=True)


async def prepare_learning_batch(trainer):
    """Prepare one useful batch; all attempts stay inside the same step timer.

    The default zero retry limit preserves strict failure behavior. The opt-in
    path is restricted to GRPO without a critic, and never calls an optimizer,
    version update, publication, profiler boundary or global-step increment.
    """
    limit = trainer.config.trainer.get("learning_acceptance_max_zero_batches", 0)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("learning_acceptance_max_zero_batches must be a nonnegative integer")
    if limit:
        estimator = trainer.config.algorithm.adv_estimator
        estimator = getattr(estimator, "value", estimator)
        if not trainer.config.trainer.get("learning_acceptance", False) or estimator != "grpo" or trainer.use_critic:
            raise ValueError("zero-signal retries require learning_acceptance and GRPO without a critic")
        trainer.metrics.update(
            {
                "learning/rejected_zero_batches": 0,
                "learning/rejected_zero_tokens": 0,
                "learning/rejected_zero_seconds": 0.0,
            }
        )

    for attempt in range(1, limit + 2):
        trainer.future_reward = None
        trainer.reward_tensor = None
        trainer.reward_extra_infos_dict = {}
        started = time.perf_counter()
        batch = await trainer._fit_generate(None)
        batch = trainer._fit_compute_reward(batch)
        batch = trainer._fit_compute_log_prob(batch)
        batch = trainer._fit_compute_ref_log_prob(batch)
        batch = trainer._fit_compute_critic(batch)
        batch = trainer._fit_compute_advantage(batch)
        if not limit:
            return batch
        mask = batch.batch["response_mask"]
        if "rollout_is_weights" in batch.batch:
            mask = mask * batch.batch["rollout_is_weights"]
        try:
            learning_batch_metrics(
                batch.batch["token_level_scores"],
                batch.batch["advantages"],
                mask,
                batch.non_tensor_batch["uid"],
                strict=True,
            )
        except LearningAcceptanceError as error:
            if not is_zero_signal_batch(error):
                trainer._log_learning_acceptance_failure(batch, error)
                raise
            seconds = time.perf_counter() - started
            will_retry = attempt <= limit
            write_rejection(trainer, batch, error, attempt, will_retry, seconds)
            if not will_retry:
                raise
            trainer.metrics["learning/rejected_zero_batches"] += 1
            trainer.metrics["learning/rejected_zero_tokens"] += error.metrics["learning/effective_tokens"]
            trainer.metrics["learning/rejected_zero_seconds"] += seconds
            batch = None
        else:
            return batch
    raise AssertionError("unreachable bounded batch preparation")
