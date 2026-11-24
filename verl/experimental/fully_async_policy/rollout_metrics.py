"""Token counts and optional trainer event traces for fully asynchronous rollouts."""

import json
import logging
import os
import time
from pathlib import Path

_files = {}
_failed = set()


def record(event, **fields):
    directory = os.environ.get("VERL_ASYNC_TRAIN_PROFILE_DIR")
    if not directory:
        return
    key = (os.getpid(), directory)
    if key in _failed:
        return
    try:
        if key not in _files:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            _files[key] = (path / f"joint-events-p{os.getpid()}.jsonl").open("x", buffering=1)
        _files[key].write(json.dumps(dict(event=event, wall_time=time.time(), **fields), separators=(",", ":")) + "\n")
    except (OSError, TypeError, ValueError):
        _failed.add(key)
        logging.getLogger(__name__).exception("Failed to write asynchronous rollout trace")


def consumed_tokens(batch):
    values = batch.batch
    response_width = values["responses"].shape[-1]
    attention = values["attention_mask"]
    total = int(attention.sum().item())
    response = int(attention[:, -response_width:].sum().item()) if response_width else 0
    return dict(
        trajectories=int(attention.shape[0]),
        prompt_tokens=total - response,
        response_tokens=response,
        total_tokens=total,
    )


def record_input(batch, round_id):
    if os.environ.get("VERL_ASYNC_TRAIN_PROFILE_DIR"):
        record("actor_rpc_input", round=int(round_id), **consumed_tokens(batch))


def required_prompt_groups(config):
    """Keep the public global batch in trajectories and verl's minibatch in prompts."""
    async_config = config.async_training
    groups = config.actor_rollout_ref.actor.ppo_mini_batch_size * async_config.require_batches
    trajectories = async_config.get("global_batch_size")
    if trajectories is None:
        return groups
    responses = config.actor_rollout_ref.rollout.n
    if type(trajectories) is not int or trajectories <= 0 or trajectories % responses:
        raise ValueError("async_training.global_batch_size must be positive and divisible by rollout.n")
    if trajectories != groups * responses:
        raise ValueError("global_batch_size must equal ppo_mini_batch_size * require_batches * rollout.n")
    if async_config.trigger_parameter_sync_step != 1:
        raise ValueError("an explicit global batch is one iteration; synchronize once after its minibatches")
    return groups
