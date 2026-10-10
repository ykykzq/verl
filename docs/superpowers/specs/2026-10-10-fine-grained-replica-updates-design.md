# Fine-Grained Replica Weight Updates and Cross-Version Migration

## Purpose

Add two capabilities to the VERL and RTP-LLM integration:

1. Rollout replicas can update weights independently, remain pinned to an older
   version, be quarantined after a failed update, and recover through a complete
   replacement update.
2. A trajectory can migrate to a replica with a different weight version by
   transferring only its token state and forcing a full prefill on the target.

The implementation must preserve the existing all-at-once update path and
same-version KVCM migration when the new options are disabled.

## Scope

The control-plane changes live in VERL. The RTP-LLM Python rollout adapter in
VERL supplies request admission, cache controls, weight-update fail-closed
behavior, and migration validation. No RTP-LLM core change is expected because
its generation configuration already exposes per-request local and remote cache
flags.

Atomic rollback to the previous weight version is out of scope. RTP-LLM updates
weights in place, and retaining a second 27B model copy is not viable on the
target hardware. Recovery therefore means keeping the replica isolated and
retrying a complete weight stream until it succeeds.

## Configuration

Add the following rollout configuration:

```yaml
actor_rollout_ref:
  rollout:
    fine_grained_weight_update:
      enabled: false
      max_retries: 1
      retry_backoff_s: 1.0
      continue_on_failure: true
    trajectory_migration:
      allow_cross_version_recompute: false
```

Both features remain disabled by default. The Qwen3.5 27B RTP-LLM recipe enables
them explicitly.

`max_retries` counts retry attempts after the initial transfer. It must be
non-negative. `retry_backoff_s` must be non-negative.

## Replica State Model

The global request router is the authoritative owner of replica lifecycle and
pin state because both routing and migration decisions already pass through it.
Each registered server has:

- `lifecycle_state`: `SERVING`, `UPDATING`, or `QUARANTINED`.
- `weight_version`: the last completely committed version.
- `desired_weight_version`: the version currently being installed, otherwise
  null.
- `pin`: null or a record containing the pinned weight version and reason.
- `last_update_error`: null after success, otherwise the final update failure.

Only `SERVING` replicas may receive new requests or become migration targets.
Pinned replicas remain `SERVING`; pinning affects weight-update eligibility, not
request admission.

Allowed transitions are:

```text
SERVING --begin update--> UPDATING
UPDATING --commit-------> SERVING
UPDATING --final fail---> QUARANTINED
QUARANTINED --recover---> UPDATING
```

A router operation atomically checks the pin and changes `SERVING` to
`UPDATING`. This prevents a concurrent pin from racing an update. Starting an
update also invalidates sticky routes and pending migrations that reference the
target replica.

Pinning captures the replica's current committed version and lasts until an
explicit unpin. Pin requests that name a different version fail closed.

## Weight Update API

`CheckpointEngineManager` exposes:

```python
async def update_weights(
    global_steps: int | None = None,
    replica_ranks: list[int] | None = None,
) -> dict: ...

async def update_replica_weights(replica_rank: int, global_steps: int) -> dict: ...
async def pin_replica(replica_rank: int, reason: str) -> dict: ...
async def unpin_replica(replica_rank: int) -> dict: ...
async def recover_replica(replica_rank: int, global_steps: int) -> dict: ...
async def get_replica_states() -> dict[int, dict]: ...
```

When fine-grained updates are disabled, `update_weights()` keeps its current
all-at-once behavior and rejects `replica_ranks`. When enabled, an omitted
`replica_ranks` selects every unpinned replica and updates them sequentially.
Explicit ranks update only the requested replicas, while still respecting pins.
`recover_replica()` is the explicit operation that may transition a quarantined
replica back to `UPDATING`.

The manager receives the router actor handle and maps `replica_rank` to the
replica's server address. A manager without a router may use the legacy update
path, but cannot enable fine-grained updates.

## Single-Replica Update Flow

The checkpoint manager serializes weight transfers with one asynchronous lock.
This is required because trainer checkpoint-engine workers cannot participate in
multiple temporary process groups concurrently.

For each selected replica:

1. Atomically ask the router to begin the update. A pinned or already-updating
   replica is skipped or rejected without touching generation.
2. Mark the replica unavailable for new routes and clear sticky routes and
   pending migration reservations that reference it.
3. Abort and drain only that replica with request rejection enabled. Aborted
   client requests retry through the router and land on another serving replica.
4. Release only that replica's reusable KV mappings.
5. Build a temporary checkpoint-engine topology containing the trainer workers
   and that replica's workers.
6. Send a complete weight stream and wait for all RTP-LLM bucket acknowledgments.
7. Finalize both sides of the temporary process group in a `finally` path.
8. On success, resume the KV allocator and generation, refresh migration
   capabilities from the server, and atomically publish the committed version
   before returning the replica to `SERVING`.

Other replicas remain in `SERVING` for the entire operation.

## Failure and Recovery

A failed bucket, validation, process-group finalization, KV resume, or generation
resume prevents the replica from returning to service. The manager:

1. Aborts any open RTP-LLM weight-update round.
2. Finalizes all checkpoint-engine participants that were prepared.
3. Leaves request admission closed.
4. Rebuilds the process group and retries a complete weight stream after
   `retry_backoff_s`.

After all attempts fail, the router records the error and transitions the
replica to `QUARANTINED`. With `continue_on_failure: true`, rolling update
continues with the remaining replicas and returns per-replica results. With it
disabled, the first final failure is raised after the replica is quarantined.

A later `recover_replica()` repeats the complete update flow. Successful
completion clears RTP-LLM's fail-closed weight error, clears the router error,
publishes the new version, and returns the replica to `SERVING`.

The result structure contains aggregate metrics plus a record for every selected
replica:

```python
{
    "replicas": {
        0: {"status": "updated", "version": 12, "attempts": 1},
        1: {"status": "pinned", "version": 11, "attempts": 0},
        2: {"status": "quarantined", "version": 10, "attempts": 2,
            "error": "..."},
    },
    "updated": 1,
    "pinned": 1,
    "quarantined": 1,
}
```

## Migration Modes

The trajectory scheduler returns one of two explicit modes:

- `remote_prefix`: source and target have the same model and weight version;
  existing KVCM domain and namespace gates apply.
- `recompute`: source and target have the same model but different weight
  versions, and `allow_cross_version_recompute` is enabled.

When cross-version recompute is disabled, the existing same-version gate remains
fail closed. A replica that is not `SERVING` cannot be selected in either mode.

### Same-Version Flow

The current KVCM flow is unchanged: wait for source remote-cache writes, issue a
verifiable prefix ticket, validate it on the target, commit the sticky route,
and resume with the full token prefix so the target can hit remote KV.

### Cross-Version Recompute Flow

For `recompute` mode:

1. Finish the current bounded generation segment on the source.
2. Transfer trajectory state only: request ID, original prompt IDs, generated
   token IDs, checkpoint index, sampling parameters, source version, target
   version, and prefix digest.
3. Do not flush, transfer, or accept KVCM state.
4. The router selects only a `SERVING` target. The target actor validates model
   identity, prefix digest, request ID, that generation admission remains open,
   and that its committed version still equals the planned target version.
5. Commit the sticky route only after target acceptance.
6. Submit the full prefix to the target with `reuse_cache=False` and
   `enable_remote_cache=False` for this request, forcing full prefill under the
   target version.
7. Continue decoding and restore normal cache behavior for later requests.

The target repeats the version check immediately before enqueue. If a weight
update occurred between acceptance and enqueue, it removes the acceptance and
returns an aborted segment. The client then replans instead of generating under
an unrecorded version.

Every successful migration history entry includes `mode`, `source_version`, and
`target_version`.

## Migration and Update Races

Beginning a replica update cancels pending migration reservations involving the
replica and invalidates sticky routes that point to it. RTP-LLM clears accepted
but not yet consumed migrations when aborting all requests. These rules cover:

- a target accepting a KVCM ticket immediately before its weights change;
- a recompute target changing version after scheduler selection;
- a source entering update while a client is between bounded segments;
- retrying an aborted segment after the original sticky target is quarantined.

The prefix token list remains the source of truth. KV cache is never reused
across weight versions.

## Observability

Expose counters and timing for:

- successful, skipped-pinned, retried, and quarantined replica updates;
- update duration and attempts per replica;
- number of serving, updating, pinned, and quarantined replicas;
- same-version and recompute migrations;
- replan events caused by target version changes;
- forced-prefill token count and duration when the backend reports them.

Router status output includes lifecycle and pin state for operational debugging.

## Compatibility

- Existing users retain all-at-once weight synchronization.
- Existing custom checkpoint managers do not need to implement the new methods.
- Existing scheduler gate configuration remains valid.
- Cross-version recompute requires RTP-LLM and text-only rollout, matching the
  current adapter's supported surface.
- Same-version KVCM migration remains the preferred path when available.

## Verification

CPU unit tests cover:

- atomic pin/update transitions and invalid pin versions;
- routing exclusion for updating and quarantined replicas;
- sticky route and migration-reservation invalidation;
- sequential single-replica updates while other replicas remain serving;
- pinned replica skipping;
- full-stream retry, final quarantine, fail-fast, and later recovery;
- scheduler mode selection with cross-version recompute enabled and disabled;
- same-version KVCM behavior regression;
- trajectory-only ticket validation and migration history;
- per-request forced cache disable and stale target-version rejection;
- accepted migration cleanup during abort.

The existing checkpoint-engine, RTP KV-cache, scheduler, and migration test
suites must remain green. Remote GPU validation must demonstrate that requests
continue on the other replicas while one replica updates, a failed replica stays
unroutable, recovery rejoins it, and a cross-version trajectory performs full
prefill before continuing.
