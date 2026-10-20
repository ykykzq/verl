# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Publication-driven placement with logical (not per-RPC) resource ownership."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trajectory:
    prompt_tokens: int
    max_new_tokens: int
    capacity_tokens: int
    server_id: str | None = None
    generated_tokens: int = 0
    weight_version: Any = None
    moved_at: float = -math.inf
    moved_tokens: int = 0
    last_probe_at: float = -math.inf
    staleness: dict = field(default_factory=dict)


class JointReplicaScheduler:
    def __init__(self, config):
        from verl.workers.config.rollout import JointSchedulingConfig

        self.config = JointSchedulingConfig(**config)
        self.clock = time.monotonic
        self.latest_version: int | None = None
        self.trajectories: dict[str, Trajectory] = {}
        self.reservations: dict[str, str] = {}
        self.pending_since: dict[str, float] = {}
        self.updated_at: dict[str, float] = {}
        self.evacuating: dict[str, float] = {}
        self.probes: dict[str, asyncio.Task] = {}
        self.probe_resources: dict[str, tuple[tuple[str, str], int]] = {}
        self.probe_errors = 0
        self.engine_limits: dict[str, dict] = {}

    @staticmethod
    def version(value):
        return value if type(value) is int and value >= 0 else -1

    def rounded_tokens(self, tokens):
        block = self.config.kv_block_size_tokens
        return ((tokens + block - 1) // block) * block

    def register(self, request_id, prompt_tokens, max_new_tokens):
        if request_id in self.trajectories:
            raise ValueError(f"trajectory {request_id!r} is already registered")
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ValueError("prompt_tokens must be a positive integer")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        budget = self.rounded_tokens(prompt_tokens + max_new_tokens)
        if budget > self.config.kv_capacity_tokens:
            raise ValueError("trajectory maximum KV requirement exceeds replica capacity")
        sequence_limits = [m.get("max_sequence_tokens") for m in self.engine_limits.values()]
        if sequence_limits and all(v is not None and prompt_tokens + max_new_tokens > v for v in sequence_limits):
            raise ValueError("trajectory exceeds every replica's sequence limit")
        self.trajectories[request_id] = Trajectory(prompt_tokens, max_new_tokens, budget)

    def finish(self, request_id):
        self.trajectories.pop(request_id, None)
        self.reservations.pop(request_id, None)
        # A Ray await cancellation is not backend completion. Running probes
        # keep their transient reservations until both scoring RPCs settle.

    def usage(self, server_id, exclude=None):
        tokens, count = 0, 0
        for rid, record in self.trajectories.items():
            if rid != exclude and record.server_id == server_id:
                tokens += record.capacity_tokens
                count += 1
        for rid, target in self.reservations.items():
            record = self.trajectories.get(rid)
            if rid != exclude and target == server_id and record is not None and record.server_id != server_id:
                tokens += record.capacity_tokens
                count += 1
        for servers, budget in self.probe_resources.values():
            if server_id in servers:
                tokens += budget
                count += 1
        return tokens, count

    def fits(self, server_id, request_id):
        tokens, count = self.usage(server_id, exclude=request_id)
        record = self.trajectories[request_id]
        sequence_limit = self.engine_limits.get(server_id, {}).get("max_sequence_tokens")
        return (
            tokens + record.capacity_tokens <= self.config.kv_capacity_tokens
            and count + 1 <= self.batch_limit(server_id)
            and (sequence_limit is None or record.prompt_tokens + record.max_new_tokens <= sequence_limit)
        )

    def batch_limit(self, server_id):
        actual = self.engine_limits.get(server_id, {}).get("max_batch_size")
        return min(self.config.max_batch_size, actual) if actual is not None else self.config.max_batch_size

    def protected(self, record, deadline=math.inf):
        if record.moved_at == -math.inf:
            return False
        return self.clock() - record.moved_at < self.config.min_decode_window_s or (
            record.generated_tokens - record.moved_tokens < self.config.min_decode_tokens
            and self.clock() < min(deadline, record.moved_at + self.config.max_defer_s)
        )

    def prepare(self, version, states):
        if type(version) is not int or version < 0:
            raise ValueError("weight version must be a nonnegative integer")
        if self.latest_version is not None and version <= self.latest_version:
            return
        self.latest_version = version
        now = self.clock()
        for sid, state in states.items():
            if self.version(state["weight_version"]) < version:
                self.pending_since.setdefault(sid, now)

    def committed_update(self, server_id):
        self.updated_at[server_id] = self.clock()
        self.evacuating.pop(server_id, None)
        # A publication can arrive while a previous transfer is in progress.
        # The router restores a pending timestamp if the committed version lags.
        self.pending_since.pop(server_id, None)

    def validate_capacity_metadata(self, metadata):
        for sid, meta in metadata.items():
            actual = meta.get("kv_block_size_tokens")
            if actual is not None and (
                type(actual) is not int or actual <= 0 or self.config.kv_block_size_tokens % actual
            ):
                raise ValueError(f"joint KV block accounting is incompatible with replica {sid}: {actual}")
            for key in ("max_sequence_tokens", "max_batch_size"):
                value = meta.get(key)
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError(f"invalid {key} capacity reported by replica {sid}")
            self.engine_limits.setdefault(sid, {}).update(
                {k: meta[k] for k in ("max_sequence_tokens", "max_batch_size") if meta.get(k) is not None}
            )

    def acquire(self, request_id, states, preferred=None):
        record = self.trajectories.get(request_id)
        if record is None:
            raise ValueError("register the logical trajectory before acquiring a replica")
        if record.server_id is not None:
            state = states.get(record.server_id, {})
            if state.get("lifecycle_state") == "SERVING":
                return record.server_id
            if state.get("lifecycle_state") == "UPDATING":
                raise RuntimeError("No available servers in load balancer")
            if record.generated_tokens:
                raise RuntimeError("Joint trajectory continuation requires migration")
            record.server_id = None
        candidates = [
            sid
            for sid, state in states.items()
            if state["lifecycle_state"] == "SERVING"
            and (self.latest_version is None or self.version(state["weight_version"]) >= self.latest_version)
            and sid not in self.evacuating
            and self.fits(sid, request_id)
        ]
        if not candidates:
            raise RuntimeError("No available servers in load balancer")
        newest = max(self.version(states[sid]["weight_version"]) for sid in candidates)
        candidates = [sid for sid in candidates if self.version(states[sid]["weight_version"]) == newest]
        sid = preferred if preferred in candidates else min(candidates, key=lambda s: (self.usage(s)[1], s))
        record.server_id = sid
        record.weight_version = states[sid]["weight_version"]
        return sid

    @staticmethod
    def compatible(source, target, same_version):
        keys = ["model_id", "rollout_config_fingerprint"]
        if same_version:
            keys += ["weight_version", "kv_transfer_domain", "kv_transfer_namespace"]
            if not all("remote_prefix" in m.get("kv_transfer_backends", []) for m in (source, target)):
                return False
        return all(source.get(key) is not None and source.get(key) == target.get(key) for key in keys)

    def select_update(self, version, states, metadata, pending_migrations):
        now = self.clock()
        candidates = []
        for sid, state in states.items():
            if state["lifecycle_state"] != "SERVING" or state.get("pin"):
                continue
            if self.version(state["weight_version"]) >= version:
                continue
            self.pending_since.setdefault(sid, now)
            if now - self.updated_at.get(sid, -math.inf) < self.config.update_cooldown_s:
                continue
            if any(
                self.protected(r, self.pending_since[sid] + self.config.max_defer_s)
                for r in self.trajectories.values()
                if r.server_id == sid
            ):
                continue
            if any(sid in (p["source_server_id"], p["target_server_id"]) for p in pending_migrations.values()):
                continue
            candidates.append(sid)
        if not candidates:
            return {"server_id": None, "reason": "cooldown, protected decode, or no eligible replica"}
        deadlines = [s for s in candidates if now - self.pending_since[s] >= self.config.max_defer_s]
        if deadlines:
            sid = min(deadlines, key=lambda s: (self.pending_since[s], self.usage(s)[1], s))
            self.evacuating.setdefault(sid, now)
            return {"server_id": sid, "reason": "maximum deferral reached"}
        empty = [s for s in candidates if not any(r.server_id == s for r in self.trajectories.values())]
        if empty:
            sid = min(empty)
            self.evacuating.setdefault(sid, now)
            return {"server_id": sid, "reason": "evacuated replica"}
        active = [s for s in candidates if s in self.evacuating]
        sid = min(active or candidates, key=lambda s: (self.usage(s)[1], self.usage(s)[0], s))
        self.evacuating.setdefault(sid, now)
        records = [(rid, r) for rid, r in self.trajectories.items() if r.server_id == sid]
        can_move = any(
            target != sid
            and target not in self.evacuating
            and state["lifecycle_state"] == "SERVING"
            and self.compatible(metadata[sid], metadata[target], True)
            and self.fits(target, rid)
            for rid, _ in records
            for target, state in states.items()
        )
        if len(records) <= self.config.sparse_request_threshold and not can_move:
            return {"server_id": sid, "reason": "sparse tail has no compatible destination"}
        if now - self.evacuating[sid] >= self.config.evacuation_timeout_s:
            return {"server_id": sid, "reason": "bounded evacuation expired"}
        return {"server_id": None, "reason": f"evacuating {sid}"}

    def target(self, request_id, source_id, source_metadata, snapshots, required):
        record = self.trajectories.get(request_id)
        if record is None:
            return None, "trajectory is no longer registered"
        if not required and self.protected(record):
            return None, "minimum continuous decode window"
        by_id = {s.server_id: s for s in snapshots}
        source = by_id.get(source_id)
        unavailable = source is None or source.lifecycle_state == "QUARANTINED"
        old_version = source_metadata.get("weight_version")
        if required and source is not None and source.lifecycle_state == "SERVING":
            if source.metadata.get("weight_version") == old_version:
                return source_id, "resume source"
        fresh = required or (
            record.staleness.get("source_version") == old_version
            and record.staleness.get("kl", -1.0) >= self.config.tail_kl_threshold
        )
        candidates = []
        for replica in snapshots:
            sid, meta = replica.server_id, replica.metadata
            if replica.lifecycle_state != "SERVING" or sid in self.evacuating or not self.fits(sid, request_id):
                continue
            if sid in self.pending_since and self.clock() + self.config.min_decode_window_s >= (
                self.pending_since[sid] + self.config.max_defer_s
            ):
                continue
            same = meta.get("weight_version") == old_version
            if not self.compatible(source_metadata, meta, same and not (required and unavailable)):
                continue
            if fresh:
                if self.version(meta.get("weight_version")) < self.version(old_version):
                    continue
                if same and not (required and unavailable):
                    continue
                if not required and meta.get("weight_version") != record.staleness.get("target_version"):
                    continue
            elif sid == source_id or not same:
                continue
            candidates.append(sid)
        if fresh and candidates:
            return min(
                candidates, key=lambda s: (-self.version(by_id[s].metadata.get("weight_version")), self.usage(s)[1], s)
            ), "staleness or version-fenced continuation"
        if source_id in self.evacuating and candidates:
            return max(candidates, key=lambda s: (self.usage(s)[1], self.usage(s)[0], s)), "same-version evacuation"
        tokens, count = self.usage(source_id)
        overloaded = (
            tokens >= self.config.kv_capacity_tokens * self.config.overload_threshold
            or count >= self.batch_limit(source_id)
        )
        candidates = [s for s in candidates if self.usage(s)[1] + 1 < count]
        if overloaded and candidates:
            return min(
                candidates, key=lambda s: (self.usage(s)[1], self.usage(s)[0], s)
            ), "same-version overload relief"
        return None, "continue current trajectory"

    def reserve(self, request_id, target_id):
        if not self.fits(target_id, request_id):
            raise RuntimeError("migration destination capacity changed")
        self.reservations[request_id] = target_id

    def commit_migration(self, request_id, target_id, resume_source=False):
        record = self.trajectories[request_id]
        record.server_id = target_id
        if not resume_source:
            record.moved_at = self.clock()
            record.moved_tokens = record.generated_tokens
            record.staleness.clear()
        self.reservations.pop(request_id, None)

    def report(self, request_id, server_id, generated_tokens, version):
        record = self.trajectories.get(request_id)
        if record is None or record.server_id != server_id:
            return None
        if (
            type(generated_tokens) is not int
            or not record.generated_tokens <= generated_tokens <= record.max_new_tokens
        ):
            raise ValueError("invalid cumulative trajectory token count")
        record.generated_tokens = generated_tokens
        record.weight_version = version
        return record

    def maybe_probe(self, request_id, prefix_ids, states, metadata, servers):
        record = self.trajectories[request_id]
        now = self.clock()
        if (
            not record.generated_tokens
            or request_id in self.probes
            or len(self.probes) >= self.config.max_concurrent_probes
            or now - record.last_probe_at < self.config.probe_interval_s
        ):
            return
        source_id = record.server_id
        source_version = record.weight_version
        source = metadata.get(source_id, {})
        if (
            states.get(source_id, {}).get("lifecycle_state") != "SERVING"
            or source.get("weight_version") != source_version
        ):
            return
        targets = [
            sid
            for sid, state in states.items()
            if state["lifecycle_state"] == "SERVING"
            and sid != source_id
            and self.version(state["weight_version"]) - self.version(source_version)
            >= self.config.staleness_version_gap
            and self.compatible(source, metadata[sid], False)
        ]
        budget = self.rounded_tokens(len(prefix_ids) + 1)

        def probe_fits(sid):
            used, count = self.usage(sid)
            limit = self.engine_limits.get(sid, {}).get("max_sequence_tokens")
            return (
                used + budget <= self.config.kv_capacity_tokens
                and count + 1 <= self.batch_limit(sid)
                and (limit is None or len(prefix_ids) + 1 <= limit)
            )

        targets = [s for s in targets if probe_fits(s)]
        if not targets or not probe_fits(source_id):
            return
        target_id = min(targets, key=lambda s: (-self.version(states[s]["weight_version"]), self.usage(s)[1], s))
        target_version = states[target_id]["weight_version"]
        tail = min(self.config.tail_tokens, record.generated_tokens, len(prefix_ids) - 1)
        record.last_probe_at = now
        self.probe_resources[request_id] = ((source_id, target_id), budget)

        async def probe():
            calls = []
            try:
                if self.trajectories.get(request_id) is not record:
                    return
                for sid, version in ((source_id, source_version), (target_id, target_version)):
                    calls.append(
                        asyncio.ensure_future(servers[sid].score_trajectory_tail.remote(prefix_ids, tail, version))
                    )
                combined = asyncio.gather(*calls, return_exceptions=True)
                responses = await asyncio.wait_for(asyncio.shield(combined), timeout=self.config.probe_timeout_s)
                if any(isinstance(result, BaseException) for result in responses):
                    raise RuntimeError("tail scoring RPC failed")
                old, new = responses
                if old["weight_version"] != source_version or new["weight_version"] != target_version:
                    return
                p, q = old["log_probs"], new["log_probs"]
                if len(p) != tail or len(q) != tail or not all(math.isfinite(x) and x <= 0 for x in p + q):
                    raise ValueError("invalid tail scoring probabilities")
                # Nonnegative sampled k3 proxy; it is not a full categorical KL.
                values = [math.expm1(b - a) - (b - a) if b - a < 700 else math.inf for a, b in zip(p, q, strict=True)]
                if (
                    self.trajectories.get(request_id) is record
                    and record.server_id == source_id
                    and record.weight_version == source_version
                    and states.get(source_id, {}).get("weight_version") == source_version
                    and states.get(target_id, {}).get("weight_version") == target_version
                    and all(states.get(s, {}).get("lifecycle_state") == "SERVING" for s in (source_id, target_id))
                ):
                    record.staleness = dict(
                        source_version=source_version,
                        target_version=target_version,
                        kl=sum(values) / tail,
                        evaluated_at=self.clock(),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.probe_errors += 1
            finally:
                if calls:
                    # Do not release GPU capacity on a local timeout: the remote
                    # RPCs have their own bounded backend timeout and drain path.
                    settled = asyncio.gather(*calls, return_exceptions=True)
                    while not settled.done():
                        try:
                            await asyncio.shield(settled)
                        except asyncio.CancelledError:
                            continue
                self.probes.pop(request_id, None)
                self.probe_resources.pop(request_id, None)

        self.probes[request_id] = asyncio.create_task(probe(), name=f"tail-staleness-{request_id}")

    def status(self):
        return dict(
            latest_version=self.latest_version,
            registered_trajectories=len(self.trajectories),
            evacuating=dict(self.evacuating),
            pending_since=dict(self.pending_since),
            active_probes=len(self.probes),
            probe_errors=self.probe_errors,
        )
