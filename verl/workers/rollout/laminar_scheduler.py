# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Laminar's same-version Repack on verl's replica and snapshot lifecycle.

See verl-project/laminar, multi_version_rollout/repack.py and
global_request_router/router.py, revision 19333d00b8d794a145a07b6c082a1fa3cf84c42e.
A Best-Fit pass can drain several sources into the same destination. The shared
publication transport installs one drained source at a time.
"""

from verl.workers.rollout.joint_scheduler import JointReplicaScheduler


class LaminarReplicaScheduler(JointReplicaScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.repack_targets: dict[str, str] = {}

    def acquire(self, request_id, states, preferred=None):
        record = self.trajectories.get(request_id)
        if record is not None and record.server_id is None and self.latest_version is not None:
            # Dispatch new work only into the published cohort. Bound requests
            # keep their existing version and continue on their original lease.
            states = {
                sid: state
                for sid, state in states.items()
                if self.version(state["weight_version"]) >= self.latest_version
            }
        return super().acquire(request_id, states, preferred)

    def select_repack(self, version, states, metadata, pending, inflight, kv_usage):
        participants = {item[key] for item in pending.values() for key in ("source_server_id", "target_server_id")}
        candidates = [
            sid
            for sid, state in states.items()
            if state["lifecycle_state"] == "SERVING"
            and not state.get("pin")
            and self.version(state["weight_version"]) < version
            and sid not in participants
        ]
        draining = self.evacuating.keys() & states.keys()
        # A settled source no longer needs its destination, even if its
        # weight install was skipped or failed.
        destinations = {
            target for source, target in self.repack_targets.items() if self.usage(source)[1] or inflight.get(source, 0)
        }
        idle = [sid for sid in candidates if sid not in destinations and not self.usage(sid)[1] and not inflight[sid]]
        if idle:
            source = min(idle, key=lambda sid: (sid not in draining, sid))
            self.evacuating.setdefault(source, self.clock())
            reason = "Laminar source drained" if source in draining else "Laminar idle replica"
            return {"server_id": source, "reason": reason}
        if draining:
            return {"server_id": None, "reason": "Laminar Repack draining"}
        if not candidates:
            return {"server_id": None, "reason": "no eligible Laminar replica"}

        assigned = self._plan_repack(candidates, metadata, kv_usage)
        if assigned:
            self.repack_targets.update(assigned)
            self.evacuating.update(dict.fromkeys(assigned, self.clock()))
            return {"server_id": None, "reason": "Laminar Repack draining", "sources": assigned}
        return {"server_id": None, "reason": "waiting for an idle replica or same-version Best-Fit"}

    def _plan_repack(self, candidates, metadata, kv_usage):
        routes = {sid: [] for sid in candidates}
        for request_id, record in self.trajectories.items():
            if record.server_id in routes:
                routes[record.server_id].append(request_id)
        # Each projected destination includes all sources already assigned to it.
        projected = {sid: (kv_usage[sid], *self.usage(sid)) for sid in candidates}
        assigned = {}
        destinations = set()
        for source in sorted(candidates, key=lambda sid: (kv_usage[sid], sid)):
            if kv_usage[source] > self.config.laminar_source_max_kv_usage:
                break
            if source in destinations or not routes[source]:
                continue
            source_kv, source_tokens, source_routes = projected[source]
            fits = []
            for target, (target_kv, target_tokens, target_routes) in projected.items():
                if target == source or not self.compatible(metadata[source], metadata[target], True):
                    continue
                kv = source_kv + target_kv
                tokens = source_tokens + target_tokens
                requests = source_routes + target_routes
                if (
                    kv <= self.config.laminar_destination_max_kv_usage
                    and tokens <= self.kv_limit(target)
                    and requests <= self.batch_limit(target)
                    and all(self.fits(target, rid) for rid in routes[source])
                ):
                    fits.append((kv, target, tokens, requests))
            if fits:
                kv, target, tokens, requests = min(fits, key=lambda entry: (-entry[0], entry[1]))
                assigned[source] = target
                destinations.add(target)
                projected[target] = (kv, tokens, requests)
                del projected[source]
        return assigned

    def committed_update(self, server_id):
        self.repack_targets.pop(server_id, None)
        super().committed_update(server_id)

    def maybe_probe(self, request_id, prefix_ids, states, metadata, servers):
        return None

    def target(self, request_id, source_id, source_metadata, snapshots, required):
        replicas = {replica.server_id: replica for replica in snapshots}
        target_id = self.repack_targets.get(source_id)
        target = replicas.get(target_id)
        if (
            target is not None
            and target.lifecycle_state == "SERVING"
            and self.compatible(source_metadata, target.metadata, True)
            and self.fits(target_id, request_id)
        ):
            return target_id, "Laminar same-version Repack"
        source = replicas.get(source_id)
        if (
            required
            and source is not None
            and source.lifecycle_state == "SERVING"
            and source.metadata.get("weight_version") == source_metadata.get("weight_version")
        ):
            return source_id, "resume source"
        return None, "retain Laminar version binding"
