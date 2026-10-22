# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Single-replica, latest-version publication independent of training workers."""

import asyncio
import json
import math
import traceback
from collections import deque


class WeightPublicationTimeoutError(asyncio.TimeoutError):
    """Publication deadline with serializable progress and no transport payloads."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics if diagnostics is not None else {}


class WeightPublicationScheduler:
    """Retain the latest immutable manifest and finish one replica at a time.

    The owner must run this on a persistent event loop (the publication actor),
    rather than the trainer's short-lived update loop. The manager owns draining,
    installation, retries and quarantine. No transfer runs in an admission RPC.
    """

    def __init__(self, manager, on_commit=None, joint_scheduling_config=None):
        self.manager = manager
        self.on_commit = on_commit
        self._joint_config = dict(joint_scheduling_config or {})
        self._joint_enabled = bool(self._joint_config.get("enabled", False))
        self._poll_interval_s = self._joint_config.get("poll_interval_s", 0.1)
        if self._joint_enabled and (
            type(self._poll_interval_s) not in (int, float)
            or not math.isfinite(self._poll_interval_s)
            or self._poll_interval_s <= 0
        ):
            raise ValueError("joint_scheduling.poll_interval_s must be finite and positive")
        self._publication_lock = asyncio.Lock()
        self._last_scheduling_decision = None
        self._latest = None
        self._active = None
        self._active_rank = None
        self._active_phase = None
        self._committed_version = -1
        self._commands = deque()
        self._wake = asyncio.Event()
        self._changed = asyncio.Condition()
        self._worker = None
        self._idle = True
        self._closing = False
        self._closed = False
        self._fatal_error = None

    @staticmethod
    def _validate_version(version):
        if type(version) is not int or version < 0:
            raise ValueError("weight version must be a nonnegative integer")

    def _check_health(self, require_open=True):
        if self._fatal_error is not None:
            raise RuntimeError(f"weight publication failed: {self._fatal_error}")
        if require_open and (self._closing or self._closed):
            raise RuntimeError("weight publication scheduler is closed or closing")

    def _start(self):
        self._idle = False
        self._wake.set()
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="weight-publication")

    async def _notify(self):
        async with self._changed:
            self._changed.notify_all()

    async def publish(self, snapshot):
        """Admit a strictly newer snapshot without waiting for any installation."""
        if self._joint_enabled:
            return await self._publish_joint(snapshot)
        self._validate_publication(snapshot)
        self._latest = snapshot
        self._start()
        await self._notify()
        return {"accepted_version": snapshot.version, "latest_version": snapshot.version}

    def _validate_publication(self, snapshot):
        self._check_health()
        self._validate_version(snapshot.version)
        if self._latest is not None and snapshot.version <= self._latest.version:
            raise ValueError("published weight versions must be strictly increasing; deduplicate before publication")

    async def _publish_joint(self, snapshot):
        async with self._publication_lock:
            self._validate_publication(snapshot)
            cancelled = False
            try:
                prepare = asyncio.ensure_future(
                    self.manager.load_balancer.prepare_weight_update.remote(snapshot.version)
                )
                # A cancelled caller cannot undo a remote admission transition.
                # Retain the snapshot and start catch-up once that RPC settles.
                while not prepare.done():
                    try:
                        await asyncio.shield(prepare)
                    except asyncio.CancelledError:
                        if prepare.cancelled():
                            raise RuntimeError("replica update preparation was cancelled") from None
                        cancelled = True
                if prepare.cancelled():
                    raise RuntimeError("replica update preparation was cancelled")
                prepare.result()
            except Exception as error:
                self._fatal_error = str(error)
                self._wake.set()
                await self._notify()
                raise
            self._latest = snapshot
            self._start()
            await self._notify()
            if cancelled:
                raise asyncio.CancelledError
            return {"accepted_version": snapshot.version, "latest_version": snapshot.version}

    async def get_status(self):
        """Return current replica progress and any observed fatal error."""
        states = await self.manager.get_replica_states()
        return {
            "latest_version": self._latest.version if self._latest is not None else None,
            "active_version": self._active.version if self._active is not None else None,
            "active_replica_rank": self._active_rank,
            "active_phase": self._active_phase,
            "committed_version": self._committed_version if self._committed_version >= 0 else None,
            "replicas": states,
            "idle": self._idle,
            "closing": self._closing,
            "closed": self._closed,
            "fatal_error": self._fatal_error,
            "scheduling_decision": self._last_scheduling_decision,
        }

    async def wait_for_version(self, version, timeout_s=300.0):
        """Wait until eligible replicas are current and a current replica serves.

        Pinned and quarantined replicas are excluded from the catch-up barrier.
        They cannot make an otherwise unavailable publication appear successful.
        A settled, unavailable publication fails immediately with recovery advice.
        """
        self._validate_version(version)
        if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("publication timeout must be finite and positive")
        clock = asyncio.get_running_loop().time
        started_at = clock()
        observed_states, observed_at = {}, None
        state_query_pending = False

        async def wait():
            nonlocal observed_states, observed_at, state_query_pending
            async with self._changed:
                while True:
                    self._check_health(require_open=False)
                    state_query_pending = True
                    states = await self.manager.get_replica_states()
                    observed_states, observed_at = states, clock()
                    state_query_pending = False
                    self._check_health(require_open=False)
                    eligible = [
                        state
                        for state in states.values()
                        if state["lifecycle_state"] != "QUARANTINED" and not state.get("pin")
                    ]
                    serving = [state for state in states.values() if state["lifecycle_state"] == "SERVING"]
                    if any(self._version(state) >= version for state in serving) and all(
                        self._version(state) >= version for state in eligible
                    ):
                        # The first-commit notification must finish before startup
                        # or validation observes publication as committed.
                        if self._active is None or self._committed_version >= version:
                            state_query_pending = True
                            return await self.get_status()
                    if self._closed:
                        raise RuntimeError(f"publication closed before version {version} became serving")
                    if self._idle and self._latest is not None and self._latest.version >= version:
                        raise RuntimeError(
                            f"no serving replica can reach weight version {version}; "
                            f"unpin or explicitly recover unavailable replicas: {states}"
                        )
                    await self._changed.wait()

        try:
            return await asyncio.wait_for(wait(), timeout=timeout_s)
        except asyncio.TimeoutError as error:
            # Reuse the last successful observation: a fresh status RPC can also hang.
            now = clock()
            latest = self._latest.version if self._latest is not None else None
            active = self._active.version if self._active is not None else None
            states = list(observed_states.values())
            diagnostics = {
                "requested_version": version,
                "timeout_s": timeout_s,
                "elapsed_s": now - started_at,
                "latest_version": latest,
                "pending_version": latest if not self._idle and (active is None or latest > active) else None,
                "committed_version": self._committed_version if self._committed_version >= 0 else None,
                "active_version": active,
                "active_replica_rank": self._active_rank,
                "phase": self._active_phase
                or ("replica_state_query" if state_query_pending else "waiting_for_publication_or_replica"),
                "state_query_pending": state_query_pending,
                "replica_states_age_s": now - observed_at if observed_at is not None else None,
                "replica_counts": {
                    "total": len(states),
                    "serving": sum(state["lifecycle_state"] == "SERVING" for state in states),
                    "serving_requested_version": sum(
                        state["lifecycle_state"] == "SERVING" and self._version(state) >= version for state in states
                    ),
                    "pending_eligible": sum(
                        state["lifecycle_state"] != "QUARANTINED"
                        and not state.get("pin")
                        and self._version(state) < version
                        for state in states
                    ),
                    "pinned": sum(bool(state.get("pin")) for state in states),
                    "quarantined": sum(state["lifecycle_state"] == "QUARANTINED" for state in states),
                },
                "replicas": {
                    rank: {
                        "lifecycle_state": state["lifecycle_state"],
                        "weight_version": state.get("weight_version"),
                        "desired_weight_version": state.get("desired_weight_version"),
                        "pinned": bool(state.get("pin")),
                    }
                    for rank, state in observed_states.items()
                },
            }
            raise WeightPublicationTimeoutError(
                f"weight publication wait timed out: {json.dumps(diagnostics, sort_keys=True)}", diagnostics
            ) from error

    async def _command(self, action, rank, reason=None):
        self._check_health()
        future = asyncio.get_running_loop().create_future()
        self._commands.append((action, rank, reason, future))
        self._start()
        await self._notify()
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def pin_replica(self, rank, reason):
        """Pin after the in-flight transfer settles, preserving its commit."""
        return await self._command("pin", rank, reason)

    async def unpin_replica(self, rank):
        """Unpin and wake catch-up against the retained latest snapshot."""
        return await self._command("unpin", rank)

    async def recover_replica(self, rank):
        """Serialize an explicit recovery using the latest manifest at execution."""
        return await self._command("recover", rank)

    @staticmethod
    def _version(state):
        version = state.get("weight_version")
        # RTP-LLM reports "initial" before its first numeric publication.
        return version if type(version) is int else -1

    async def _transfer(self, snapshot, rank, recovery=False):
        self._active = snapshot
        self._active_rank = rank
        self._active_phase = "replica_update"
        await self._notify()
        try:
            result = await self.manager.update_replica_from_snapshot(snapshot, rank, recovery=recovery)
            entry = result["replicas"][rank]
            if entry["status"] == "updated" and snapshot.version > self._committed_version:
                if self.on_commit is not None:
                    self._active_phase = "commit_callback"
                    await self._notify()
                    await self.on_commit(snapshot.version, result)
                self._committed_version = snapshot.version
            return result
        except Exception as error:
            self._fatal_error = str(error)
            raise
        finally:
            self._active = None
            self._active_rank = None
            self._active_phase = None
            await self._notify()

    async def _execute_command(self, action, rank, reason, future):
        try:
            if action == "pin":
                result = await self.manager.pin_replica(rank, reason)
            elif action == "unpin":
                result = await self.manager.unpin_replica(rank)
            else:
                states = await self.manager.get_replica_states()
                if rank not in states:
                    raise ValueError(f"unknown replica rank {rank}")
                if self._latest is None:
                    raise ValueError("recovery requires a published weight snapshot")
                state = states[rank]
                if state["lifecycle_state"] != "QUARANTINED":
                    raise RuntimeError(f"replica {rank} is {state['lifecycle_state']}; recovery requires QUARANTINED")
                # Errors from a transfer or its commit callback are fatal even
                # when initiated through an explicit recovery RPC.
                result = await self._transfer(self._latest, rank, recovery=True)
        except asyncio.CancelledError:
            if not future.done():
                future.set_exception(RuntimeError("weight publication scheduler closed during operation"))
            raise
        except Exception as error:
            if not future.done():
                future.set_exception(error)
            if self._fatal_error is not None:
                raise
        else:
            if not future.done():
                future.set_result(result)
        finally:
            await self._notify()

    async def _run(self):
        try:
            while True:
                self._check_health(require_open=False)
                self._wake.clear()
                while self._commands:
                    command = self._commands.popleft()
                    await self._execute_command(*command)
                    command = None
                states = await self.manager.get_replica_states()
                snapshot = self._latest
                candidates = (
                    []
                    if snapshot is None
                    else [
                        rank
                        for rank, state in states.items()
                        if state["lifecycle_state"] == "SERVING"
                        and not state.get("pin")
                        and self._version(state) < snapshot.version
                    ]
                )
                if candidates:
                    if self._joint_enabled:
                        decision = await self.manager.load_balancer.select_replica_for_update.remote(snapshot.version)
                        self._last_scheduling_decision = decision
                        if snapshot is not self._latest:
                            snapshot = None
                            continue
                        server_id = decision.get("server_id")
                        ranks = {replica.server_address: replica.replica_rank for replica in self.manager.replicas}
                        if server_id is not None and server_id not in ranks:
                            raise RuntimeError(f"joint scheduler selected unknown rollout replica {server_id!r}")
                        rank = ranks.get(server_id)
                        if rank is None or rank not in candidates:
                            snapshot = None
                            # Deferred work is not idle: wait_for_version must keep
                            # waiting, and close(drain=True) must finish catch-up.
                            self._idle = False
                            try:
                                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_interval_s)
                            except asyncio.TimeoutError:
                                pass
                            continue
                    else:
                        rank = min(candidates, key=lambda rank: (self._version(states[rank]), rank))
                    await self._transfer(snapshot, rank)
                    snapshot = None
                    continue
                snapshot = None  # Do not keep superseded object refs while idle.
                if self._wake.is_set():
                    continue
                self._idle = True
                await self._notify()
                if self._closing:
                    return
                await self._wake.wait()
                self._idle = False
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Retain the message rather than a traceback that can retain old
            # tensor manifests and object-store references indefinitely.
            self._fatal_error = str(error)
        finally:
            self._idle = True
            self._settle_commands()
            await self._notify()

    def _settle_commands(self):
        while self._commands:
            _, _, _, future = self._commands.popleft()
            if not future.done():
                future.set_exception(RuntimeError(self._fatal_error or "weight publication scheduler closed"))

    async def close(self, drain=True):
        """Stop admission, drain or cancel transfers, and release all manifests."""
        # Serialize against the short preparation RPC, never a weight transfer.
        async with self._publication_lock:
            self._closing = True
        self._wake.set()
        try:
            if self._worker is not None:
                if not drain and not self._worker.done():
                    self._worker.cancel()
                try:
                    await self._worker
                except asyncio.CancelledError as error:
                    if drain:
                        raise
                    # Python 3.10 can cache the original traceback separately
                    # and chain its cancellation exception as the context.
                    # Clear finished frames, including chained cleanup errors.
                    pending = [error]
                    seen = set()
                    while pending:
                        error = pending.pop()
                        if id(error) in seen:
                            continue
                        seen.add(id(error))
                        traceback.clear_frames(error.__traceback__)
                        error.__traceback__ = None
                        pending.extend(e for e in (error.__cause__, error.__context__) if e is not None)
        finally:
            self._settle_commands()
            self._worker = None
            self._latest = None
            self._active = None
            self._active_rank = None
            self._closed = True
            await self._notify()
        self._check_health(require_open=False)
