"""Validate native profile drain evidence after explicit profiling freeze.

Pure Python: no CUDA synchronization or native API is added by this module.
The native process may append its final destructor footer later; the SHA here
identifies the exact post-seal prefix, not an immutable closed file. The native
engine remains alive with admission closed; TP engine stop is not used.
"""

import asyncio
import hashlib
import json
import math
from pathlib import Path
import time


def _read_lines(path):
    with path.open("rb") as stream:
        yield from stream


def seal_native_profile(directory, *, host, pid, replica_rank, tp_rank, seal_requested_us):
    root = Path(directory)
    paths = sorted(root.glob(f"request-profile-{pid}-*.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one native profile for pid {pid}, found {len(paths)}")
    path = paths[0]
    digest = hashlib.sha256()
    first = last = None
    count = size = forwards = ends = 0
    for line in _read_lines(path):
        digest.update(line)
        size += len(line)
        if not line.endswith(b"\n") or not line.strip():
            raise RuntimeError("native profile contains a partial or empty record")
        row = json.loads(line)
        count += 1
        first = row if first is None else first
        last = row
        if (row.get("host"), row.get("pid"), row.get("replica_rank"), row.get("tp_rank")) != (
            host,
            pid,
            replica_rank,
            tp_rank,
        ):
            raise RuntimeError("native profile actor identity differs from sealing actor")
        if row.get("process_tp_rank") != tp_rank:
            raise RuntimeError("native profile process TP identity differs")
        event = row.get("event")
        if event == "profile_begin" and count != 1:
            raise RuntimeError("native profile contains multiple begin records")
        if event in {"request_phase_error", "prefill_profile_error", "profile_after_seal_error"}:
            raise RuntimeError(f"native profile recorded {event}")
        if event == "request_phase" and row.get("timing_valid") is not True:
            raise RuntimeError("native profile recorded invalid request timing")
        if event == "prefill_forward":
            if ends:
                raise RuntimeError("native profile records a forward after its drain marker")
            value = row.get("forward_cuda_ms")
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or not row.get("forward_ok"):
                raise RuntimeError("native profile recorded failed or invalid CUDA forward")
            forwards += 1
        if event == "prefill_profile_end":
            ends += 1
            if row.get("invalid_batches") != 0 or row.get("completed_batches") != forwards:
                raise RuntimeError("native prefill drain counters do not match forward records")
        if event in {"profile_status", "profile_end"}:
            if row.get("dropped_records") != 0 or row.get("io_or_queue_failure") is not False:
                raise RuntimeError("native profile reports dropped records or writer failure")
            if row.get("accepted_records") != count - 1 or row.get("written_records") != count - 1:
                raise RuntimeError("native profile record counters do not match its file")
    if first is None or first.get("event") != "profile_begin" or ends != 1:
        raise RuntimeError("native profile is missing its unique begin/drain evidence")
    if last.get("event") != "profile_status" or last.get("reason") != "explicit_reprefill_seal":
        raise RuntimeError("native profile is not sealed by the production executor")
    if last.get("active_request_phases") != 0:
        raise RuntimeError("native profile sealed with outstanding RPC phase records")
    if last.get("record_us", 0) < seal_requested_us:
        raise RuntimeError("native profile seal marker predates this finalization")
    result = dict(
        enabled=True,
        host=host,
        pid=pid,
        replica_rank=replica_rank,
        tp_rank=tp_rank,
        path=str(path),
        prefix_bytes=size,
        prefix_sha256=digest.hexdigest(),
        records=count,
        forward_batches=forwards,
        native_engine_stopped=False,
        native_profile_sealed=True,
        no_drops=True,
        seal_requested_us=seal_requested_us,
        seal_record_us=last["record_us"],
        sealed_us=time.time_ns() // 1000,
        native_destructor_footer_may_append=True,
    )
    destination = root / f"profile-seal-{host}-{pid}.json"
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    result["seal_path"] = str(destination)
    return result


async def _settle_actor_calls(refs, timeout_s, phase):
    tasks = [asyncio.ensure_future(ref) for ref in refs]
    errors = []

    def observe(task):
        if not task.cancelled() and task.exception() is not None:
            errors.append(task.exception())

    for task in tasks:
        task.add_done_callback(observe)
    group = asyncio.gather(*tasks, return_exceptions=True)
    try:
        # Timeout/caller cancellation may stop waiting but must not cancel a
        # rank's native seal. Cached actor tasks make a retry idempotent.
        results = await asyncio.wait_for(asyncio.shield(group), timeout_s)
    except (asyncio.TimeoutError, asyncio.CancelledError) as error:
        if errors:
            raise error from errors[0]
        raise
    if errors:
        raise RuntimeError(f"{len(errors)} RTP profile {phase}(s) failed") from errors[0]
    return results


async def finalize_server_profiles(servers, timeout_s=240):
    """Drain every actor first, then submit every TP seal before awaiting any."""
    if not servers:
        raise RuntimeError("profiling enabled but no RTP server actors were found")
    prepared = await _settle_actor_calls(
        [server.prepare_reprefill_profile_shutdown.remote() for server in servers], timeout_s, "preparation"
    )
    if any(not isinstance(result, dict) or not result.get("ready") for result in prepared):
        raise RuntimeError("RTP profile preparation did not confirm every actor drained")
    results = await _settle_actor_calls(
        [server.finalize_reprefill_profile.remote() for server in servers], timeout_s, "finalization"
    )
    identities = []
    for result in results:
        if (
            not isinstance(result, dict)
            or not result.get("enabled")
            or not result.get("native_profile_sealed")
            or not result.get("no_drops")
        ):
            raise RuntimeError("RTP server did not return a verified native profile seal")
        identities.append((result["replica_rank"], result["tp_rank"]))
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate replica/TP actors in native profile finalization")
    return results
