"""Opt-in evidence for a useful GRPO batch and an actual optimizer update."""

import math
from collections import defaultdict

import torch


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
    metrics = {
        "learning/groups_with_reward_variance": varied,
        "learning/group_reward_variance_max": max(variances, default=0.0),
        "learning/effective_tokens": int(valid.sum()),
        "learning/nonzero_advantage_tokens": effective,
        "learning/finite": float(finite),
    }
    if strict:
        if not finite:
            raise ValueError("learning acceptance: nonfinite rewards or effective advantages")
        if not varied:
            raise ValueError("learning acceptance: no nonzero within-group reward variance")
        if not effective:
            raise ValueError("learning acceptance: no nonzero effective advantage")
    return metrics


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
