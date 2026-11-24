"""Identity snapshots for unquantized TP1 vLLM workers.

The trainer exports HF named tensors. vLLM's model loader performs packing at
installation, so applying RTP's tensor conversion here would corrupt weights.
This plan plugs into the existing immutable, sharded NCCL snapshot transport.
"""

import hashlib
import json
import math
import time
from copy import deepcopy


_DTYPE_BYTES = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _layout(value):
    result = {}
    for name, shape, dtype in value:
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("snapshot tensor names must be unique nonempty strings")
        if dtype not in _DTYPE_BYTES or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f"invalid HF tensor layout: {name}")
        result[name] = (tuple(shape), dtype)
    if not result:
        raise ValueError("empty HF snapshot layout")
    return result


class HFSnapshotPlan:
    kind = "hf_identity_tp1_v1"

    def __init__(self, source_layout, target_layout, transport):
        started = time.perf_counter()
        if transport.get("tp_rank") != 0 or transport.get("tp_size") != 1:
            raise ValueError("HF identity snapshots require TP1")
        target = _layout(target_layout)
        sources = _layout([(x["name"], x["shape"], x["dtype"]) for x in source_layout])
        if sources != target:
            missing = sorted(target.keys() - sources.keys())
            extra = sorted(sources.keys() - target.keys())
            changed = sorted(n for n in sources.keys() & target.keys() if sources[n] != target[n])
            raise ValueError(f"HF snapshot layout mismatch: missing={missing}, extra={extra}, changed={changed}")
        self.required_names = frozenset(target)
        self.ignored_source_names = frozenset()
        largest = max(math.prod(s) * _DTYPE_BYTES[d] for s, d in target.values())
        self.buffer_budgets = {
            # No transformed tensor is allocated by this identity plan. The
            # full-tensor export remains owned until the NCCL transfer settles.
            "conversion_peak_bytes": 0,
            "largest_source_tensor_bytes": largest,
        }
        self.manifest = {
            "kind": self.kind,
            "source_layout": deepcopy(source_layout),
            "target_layout": [[n, list(s), d] for n, (s, d) in sorted(target.items())],
            "transport": deepcopy(transport),
            "buffer_budgets": self.buffer_budgets,
        }
        self.plan_hash = hashlib.sha256(_canonical(self.manifest)).hexdigest()
        self._target = target
        self.build_seconds = time.perf_counter() - started

    def export(self):
        return deepcopy(self.manifest)

    @classmethod
    def restore(cls, payload, expected_hash):
        if not isinstance(payload, dict) or payload.get("kind") != cls.kind:
            raise ValueError("not an HF identity snapshot plan")
        if hashlib.sha256(_canonical(payload)).hexdigest() != expected_hash:
            raise ValueError("HF snapshot manifest hash mismatch")
        plan = cls(payload["source_layout"], payload["target_layout"], payload["transport"])
        if plan.plan_hash != expected_hash:
            raise ValueError("HF snapshot manifest is not canonical")
        return plan

    def new_round(self, version, device, max_pending_bytes):
        if type(version) is not int or version < 0:
            raise ValueError("snapshot version must be a nonnegative integer")
        if getattr(device, "type", None) != "cuda":
            raise ValueError("HF runtime snapshots require CUDA tensors")
        return HFSnapshotRound(self, version, device)


class HFSnapshotRound:
    def __init__(self, plan, version, device):
        self.plan, self.version, self.device = plan, version, device
        self.seen = set()
        self.finished = self.closed = False

    def add(self, weights, is_last=False):
        if self.closed or self.finished:
            raise RuntimeError("HF snapshot round is no longer writable")
        outputs = []
        for name, tensor in weights:
            if name not in self.plan._target or name in self.seen:
                raise ValueError(f"unexpected or repeated HF snapshot tensor: {name}")
            shape, dtype = self.plan._target[name]
            if tuple(tensor.shape) != shape or str(tensor.dtype) != dtype:
                raise ValueError(f"HF snapshot tensor layout mismatch: {name}")
            if tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError(f"HF snapshot tensor requires contiguous local CUDA storage: {name}")
            self.seen.add(name)
            outputs.append((name, tensor))
        if is_last:
            missing = self.plan.required_names - self.seen
            if missing:
                raise ValueError(f"incomplete HF snapshot: {sorted(missing)}")
            self.finished = True
        return outputs

    def close(self):
        self.closed = True


def describe_hf_layout(model_path, dtype):
    """Build only metadata, without allocating a second copy of model weights."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    if config.model_type != "qwen2" or config.tie_word_embeddings:
        raise ValueError("V17 identity snapshot validation is restricted to untied Qwen2 models")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    name_dtype = dtype if str(dtype).startswith("torch.") else "torch." + str(dtype)
    layout = [[n, list(p.shape), name_dtype] for n, p in model.named_parameters()]
    _layout(layout)
    return {"layout": layout, "required_names": sorted(n for n, _, _ in layout), "ignored_source_names": []}
