"""Opt-in evidence for a useful GRPO batch and an actual optimizer update."""

import math
from collections import defaultdict
from itertools import islice

import numpy as np
import torch


class LearningAcceptanceError(ValueError):
    """Strict acceptance failure whose evidence also survives exception serialization."""

    def __init__(self, message, metrics=None, diagnostics=None):
        super().__init__(message)
        self.metrics = metrics if metrics is not None else {}
        self.diagnostics = diagnostics if diagnostics is not None else {}


def learning_batch_metrics(rewards, advantages, mask, group_ids, *, strict=False):
    if rewards.shape != advantages.shape or mask.shape != advantages.shape or len(group_ids) != rewards.shape[0]:
        raise ValueError("learning acceptance batch shapes do not match")
    valid = mask > 0
    if not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("learning acceptance requires a finite nonnegative token mask")
    scores = rewards.sum(-1).detach().cpu()
    groups = defaultdict(list)
    for group, score in zip(group_ids, scores, strict=True):
        groups[str(group)].append(float(score))
    variances = [
        torch.tensor(values, dtype=torch.float64).var(unbiased=False).item()
        for values in groups.values()
        if len(values) > 1
    ]
    finite = bool(torch.isfinite(scores).all() and torch.isfinite(advantages[valid]).all())
    varied = sum(math.isfinite(value) and value > 0 for value in variances)
    effective = int(((advantages != 0) & valid).sum())
    group_sizes = [len(values) for values in groups.values()]
    metrics = {
        "learning/uid_groups": len(groups),
        "learning/singleton_uid_groups": sum(size == 1 for size in group_sizes),
        "learning/multisample_uid_groups": len(variances),
        "learning/uid_group_size_min": min(group_sizes, default=0),
        "learning/uid_group_size_max": max(group_sizes, default=0),
        "learning/groups_with_reward_variance": varied,
        "learning/group_reward_variance_max": max(variances, default=0.0),
        "learning/effective_tokens": int(valid.sum()),
        "learning/nonzero_advantage_tokens": effective,
        "learning/finite": float(finite),
    }
    failure = None
    if strict:
        if not finite:
            failure = "nonfinite rewards or effective advantages"
        elif not variances:
            failure = "no multi-sample UID groups; cannot assess within-group reward variance"
        elif not varied:
            failure = "no nonzero within-group reward variance; all multi-sample groups have constant rewards"
        elif not effective:
            failure = "no nonzero effective advantage"
    if failure:
        sample_count = min(8, len(group_ids))
        diagnostics = {
            "metrics": {key: _bounded_evidence(value) for key, value in metrics.items()},
            "groups": [
                {
                    "uid": str(uid)[:1024],
                    "sample_count": len(values),
                    "rewards": [_bounded_evidence(value) for value in values[:16]],
                    "rewards_omitted": max(0, len(values) - 16),
                }
                for uid, values in islice(groups.items(), 32)
            ],
            "groups_omitted": max(0, len(groups) - 32),
            "samples": [
                {
                    "index": index,
                    "uid": str(group_ids[index])[:1024],
                    "reward": _bounded_evidence(float(scores[index])),
                    "effective_tokens": int(valid[index].sum()),
                    "nonzero_advantage_tokens": int(((advantages[index] != 0) & valid[index]).sum()),
                }
                for index in range(sample_count)
            ],
            "samples_omitted": len(group_ids) - sample_count,
        }
        raise LearningAcceptanceError(f"learning acceptance: {failure}", metrics, diagnostics)
    return metrics


def _bounded_evidence(value, depth=0):
    """Keep auxiliary scoring metadata small and valid for strict JSON writers."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value if len(value) <= 1024 else value[:510] + " ... " + value[-509:]
    if depth >= 2:
        return _bounded_evidence(str(value))
    if isinstance(value, dict):
        return {str(key)[:128]: _bounded_evidence(item, depth + 1) for key, item in islice(value.items(), 16)}
    if isinstance(value, np.ndarray | torch.Tensor):
        if value.ndim == 0:
            return _bounded_evidence(value.item(), depth)
        value = value[:16].tolist()
    if isinstance(value, list | tuple):
        return [_bounded_evidence(item, depth + 1) for item in value[:16]]
    return _bounded_evidence(str(value))


def add_learning_batch_evidence(diagnostics, batch, tokenizer, reward_extra_infos, response_length_limit):
    """Enrich the bounded failure samples without confusing rejected tokens with padding."""
    responses = batch.batch["responses"]
    response_width = responses.shape[-1]
    attention_mask = batch.batch.get("attention_mask")
    response_mask = attention_mask[:, -response_width:] if attention_mask is not None else batch.batch["response_mask"]
    lengths = (response_mask > 0).sum(-1)
    diagnostics["response_length"] = {
        "min": int(lengths.min()) if lengths.numel() else 0,
        "max": int(lengths.max()) if lengths.numel() else 0,
        "mean": float(lengths.float().mean()) if lengths.numel() else 0.0,
        "limit": response_length_limit,
    }
    eos_ids = getattr(tokenizer, "eos_token_id", None)
    if eos_ids is not None and not isinstance(eos_ids, list | tuple | set):
        eos_ids = [eos_ids]
    for sample in diagnostics["samples"]:
        index = sample["index"]
        token_ids = responses[index][response_mask[index] > 0].detach().cpu().tolist()
        ended_with_eos = bool(token_ids and token_ids[-1] in eos_ids) if eos_ids is not None else None
        reached_limit = len(token_ids) >= response_length_limit if response_length_limit is not None else None
        selected_ids = token_ids if len(token_ids) <= 128 else token_ids[:64] + token_ids[-64:]
        sample.update(
            {
                "response_length": len(token_ids),
                "ended_with_eos": ended_with_eos,
                "reached_response_limit": reached_limit,
                "possible_length_truncation": reached_limit and not ended_with_eos,
                "response_token_ids": selected_ids,
                "response_tokens_omitted": max(0, len(token_ids) - 128),
                "response_text": _bounded_evidence(tokenizer.decode(selected_ids, skip_special_tokens=False)),
                "reward_evidence": {
                    str(key)[:128]: _bounded_evidence(values[index])
                    for key, values in islice(reward_extra_infos.items(), 16)
                },
            }
        )
        for key in ("stop_reason", "finish_reason", "data_source"):
            values = batch.non_tensor_batch.get(key)
            sample[key] = _bounded_evidence(values[index]) if values is not None else None
        reward_models = batch.non_tensor_batch.get("reward_model")
        if reward_models is not None and isinstance(reward_models[index], dict):
            sample["ground_truth"] = _bounded_evidence(reward_models[index].get("ground_truth"))


class ParameterUpdateProbe:
    """Bounded CPU snapshots of local parameter shards; only used in acceptance runs."""

    def __init__(self, parameters, max_elements_per_tensor=4096):
        if max_elements_per_tensor <= 0:
            raise ValueError("parameter probe size must be positive")
        self.samples = []
        self.device = None
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            local = self.local(parameter)
            self.device = local.device
            if local.numel():
                stride = max(1, math.ceil(local.numel() / max_elements_per_tensor))
                before = local.detach().reshape(-1)[::stride].to("cpu", copy=True)
                self.samples.append((parameter, stride, before))

    @staticmethod
    def local(parameter):
        return parameter.to_local() if hasattr(parameter, "to_local") else parameter

    def finish(self, grad_norm):
        delta_max, checked = 0.0, 0
        invalid = not math.isfinite(grad_norm) or grad_norm <= 0
        device = self.device
        for parameter, stride, before in self.samples:
            local = self.local(parameter)
            device = local.device
            after = local.detach().reshape(-1)[::stride].cpu()
            invalid |= not bool(torch.isfinite(before).all() and torch.isfinite(after).all())
            delta_max = max(delta_max, (after.double() - before.double()).abs().max().item())
            checked += after.numel()
        evidence = torch.tensor([float(invalid), delta_max, checked], dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(evidence, op=torch.distributed.ReduceOp.MAX)
        invalid, delta_max, checked = evidence.tolist()
        if invalid:
            raise ValueError("learning acceptance: gradient must be finite and nonzero, parameters must be finite")
        if checked == 0 or delta_max <= 0:
            raise ValueError("learning acceptance: sampled parameters did not change after optimizer step")
        return {"parameter_delta_max": delta_max, "parameter_probe_elements_per_rank_max": checked}
