# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Single-replica, latest-version publication independent of training workers."""

import asyncio
import math
from collections import deque


class WeightPublicationScheduler:
    """Retain the latest immutable manifest and finish one replica at a time.

    The owner must run this on a persistent event loop (the publication actor),
    rather than the trainer's short-lived update loop. The manager owns draining,
    installation, retries and quarantine. No transfer runs in an admission RPC.
    """

    def __init__(self, manager, on_commit=None):
        self.manager = manager
        self.on_commit = on_commit
        self._latest = None
        self._active = None
        self._active_rank = None
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
        self._check_health()
        self._validate_version(snapshot.version)
        if self._latest is not None and snapshot.version <= self._latest.version:
            raise ValueError("published weight versions must be strictly increasing; deduplicate before publication")
        self._latest = snapshot
        self._start()
        await self._notify()
        return {"accepted_version": snapshot.version, "latest_version": snapshot.version}

    async def get_status(self):
        """Return current replica progress and any observed fatal error."""
        states = await self.manager.get_replica_states()
        return {
            "latest_version": self._latest.version if self._latest is not None else None,
            "active_version": self._active.version if self._active is not None else None,
            "active_replica_rank": self._active_rank,
            "committed_version": self._committed_version if self._committed_version >= 0 else None,
            "replicas": states,
            "idle": self._idle,
            "closing": self._closing,
            "closed": self._closed,
            "fatal_error": self._fatal_error,
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

        async def wait():
            async with self._changed:
                while True:
                    self._check_health(require_open=False)
                    states = await self.manager.get_replica_states()
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
                            return await self.get_status()
                    if self._closed:
                        raise RuntimeError(f"publication closed before version {version} became serving")
                    if self._idle and self._latest is not None and self._latest.version >= version:
                        raise RuntimeError(
                            f"no serving replica can reach weight version {version}; "
                            f"unpin or explicitly recover unavailable replicas: {states}"
                        )
                    await self._changed.wait()

        return await asyncio.wait_for(wait(), timeout=timeout_s)

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
        await self._notify()
        try:
            result = await self.manager.update_replica_from_snapshot(snapshot, rank, recovery=recovery)
            entry = result["replicas"][rank]
            if entry["status"] == "updated" and snapshot.version > self._committed_version:
                if self.on_commit is not None:
                    await self.on_commit(snapshot.version, result)
                self._committed_version = snapshot.version
            return result
        except Exception as error:
            self._fatal_error = str(error)
            raise
        finally:
            self._active = None
            self._active_rank = None
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
                    # A cancelled Task caches its cancellation exception. Clear
                    # transfer frames that otherwise keep object refs alive.
                    error.__traceback__ = None
        finally:
            self._settle_commands()
            self._worker = None
            self._latest = None
            self._active = None
            self._active_rank = None
            self._closed = True
            await self._notify()
        self._check_health(require_open=False)
