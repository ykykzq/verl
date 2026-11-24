"""Observe executed prefix tokens and CUDA forward time without a per-step sync.

Mixed prefill/decode batches share kernels: their GPU duration is measured,
while token-proportional attribution is explicitly an estimate, not an isolated
re-prefill kernel measurement. Server events account for requests never run.
"""

import json
import os
import time
import threading
from collections import deque
from pathlib import Path


class PrefixAccounting:
    def __init__(self):
        self.requests = {}

    def observe(self, output):
        # Scheduler sends positions BEFORE this forward. It may split a long
        # prefill into several chunks or resume a preempted request from zero.
        for rid in output.finished_req_ids:
            self.requests.pop(rid, None)
        starts = {}
        for req in output.scheduled_new_reqs:
            extra = getattr(req.sampling_params, "extra_args", None) or {}
            meta = dict(extra.get("v17_profile") or {})
            state = self.requests.get(req.req_id)
            if state is None:
                length = len(req.prompt_token_ids or [])
                generated = int(meta.get("generated_before", 0))
                original = int(meta.get("original_prompt_tokens", length))
                if generated < 0 or original < 0 or original + generated != length:
                    raise ValueError("profile prompt and retained-prefix lengths disagree")
                state = dict(
                    meta=meta,
                    prompt_tokens=length,
                    frontier=0,
                    next_position=0,
                    episode=0,
                    replay_until=max(0, length - 1) if generated else 0,
                    counted=False,
                )
                self.requests[req.req_id] = state
            starts[req.req_id] = req.num_computed_tokens
        cached = output.scheduled_cached_reqs
        if len(cached.req_ids) != len(cached.num_computed_tokens):
            raise ValueError("cached request profile positions are incomplete")
        starts.update(zip(cached.req_ids, cached.num_computed_tokens))
        rows = []
        for rid, count in output.num_scheduled_tokens.items():
            if rid not in starts or rid not in self.requests:
                raise RuntimeError(f"missing scheduler profile metadata for {rid}")
            state = self.requests[rid]
            begin = starts[rid]
            if begin < state["next_position"]:
                state["episode"] += 1
                state["replay_until"] = max(state["replay_until"], state["frontier"])
                state["counted"] = False
            end = begin + count
            repeat = max(0, min(end, state["replay_until"]) - begin)
            prompt = max(0, min(end, state["prompt_tokens"]) - begin)
            first_replay = repeat > 0 and not state["counted"]
            if repeat:
                state["counted"] = True
            state["frontier"] = max(state["frontier"], end)
            state["next_position"] = end
            rows.append(
                dict(
                    request_id=rid,
                    **state["meta"],
                    episode=state["episode"],
                    position=begin,
                    scheduled_tokens=count,
                    prompt_tokens_computed=prompt,
                    reprefill_tokens=repeat,
                    reprefill_episode_start=first_replay,
                )
            )
        return rows


class CUDAForwardProfile:
    def __init__(self, directory, replica_rank, device, backend="vllm"):
        import torch

        self.lock = threading.RLock()
        self.torch, self.device = torch, device
        self.accounting = PrefixAccounting()
        self.pending = deque()
        self.sequence = 0
        self.sealed = False
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / f"{backend}-native-r{replica_rank}-p{os.getpid()}.jsonl"
        self.file = self.path.open("x", buffering=1)
        self.totals = dict(
            forwards=0,
            gpu_ms=0.0,
            scheduled_tokens=0,
            reprefill_tokens=0,
            reprefill_episodes=0,
            mixed_reprefill_batch_gpu_ms=0.0,
            token_share_gpu_ms_estimate=0.0,
        )
        self.write(
            dict(
                event="profile_open",
                schema=1,
                pid=os.getpid(),
                replica_rank=replica_rank,
                clock="CUDA events on execution stream; wall timestamp at enqueue",
                attribution="token_share_gpu_ms_estimate is not isolated kernel time",
            )
        )

    def write(self, value):
        self.file.write(json.dumps(value, separators=(",", ":")) + "\n")

    def start(self, output):
        return self.start_rows(self.accounting.observe(output), output.total_num_scheduled_tokens)

    def start_rows(self, rows, scheduled_tokens):
        with self.lock:
            return self._start_rows(rows, scheduled_tokens)

    def _start_rows(self, rows, scheduled_tokens):
        if self.sealed:
            raise RuntimeError("cannot execute after profile seal")
        self.drain()
        if scheduled_tokens == 0:
            return None
        if sum(r["scheduled_tokens"] for r in rows) != scheduled_tokens:
            raise ValueError("profile row total differs from executed batch")
        if len(self.pending) >= 2048:
            raise RuntimeError("CUDA profile completion queue overflow; evidence would be incomplete")
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        record = dict(sequence=self.sequence, wall_start=time.time(), rows=rows, scheduled_tokens=scheduled_tokens)
        self.sequence += 1
        start.record(self.torch.cuda.current_stream(self.device))
        return start, end, record

    def finish(self, ticket, failed=False):
        if ticket is None:
            return
        start, end, record = ticket
        end.record(self.torch.cuda.current_stream(self.device))
        record["failed"] = failed
        with self.lock:
            self.pending.append((start, end, record))

    def drain(self, synchronize=False):
        if synchronize and self.pending:
            self.pending[-1][1].synchronize()
        while self.pending and self.pending[0][1].query():
            start, end, record = self.pending.popleft()
            duration = float(start.elapsed_time(end))
            rows = record.pop("rows")
            replay = [r for r in rows if r["reprefill_tokens"]]
            repeated = sum(r["reprefill_tokens"] for r in replay)
            share = duration * repeated / record["scheduled_tokens"]
            self.totals["forwards"] += 1
            self.totals["gpu_ms"] += duration
            self.totals["scheduled_tokens"] += record["scheduled_tokens"]
            self.totals["reprefill_tokens"] += repeated
            self.totals["reprefill_episodes"] += sum(r["reprefill_episode_start"] for r in replay)
            self.totals["mixed_reprefill_batch_gpu_ms"] += duration if repeated else 0
            self.totals["token_share_gpu_ms_estimate"] += share
            # Keep small batch durations for timeline and all replay identities;
            # ordinary decode requests do not produce per-token JSON records.
            self.write(
                dict(
                    event="forward",
                    **record,
                    gpu_ms=duration,
                    reprefill_tokens=repeated,
                    replay_requests=replay,
                    token_share_gpu_ms_estimate=share,
                )
            )

    def seal(self):
        if not self.sealed:
            self.drain(synchronize=True)
            self.write(dict(event="profile_sealed", wall_time=time.time(), totals=self.totals))
            self.file.close()
            self.sealed = True
        return dict(path=str(self.path), sealed=True, totals=dict(self.totals))
