# Independent Weight Publication Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement and review the following tasks.

**Goal:** Move replica weight publication off training workers after a complete,
immutable snapshot has been exported.

**Architecture:** CPU snapshots in Ray's object store feed a separate publication
actor. A bounded latest-version scheduler reuses replica lifecycle handling and
the existing RTP-LLM IPC consumer without trainer-side transfer collectives.

**Tech Stack:** Python, asyncio, PyTorch, Ray, OmegaConf, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-independent-weight-publication-design.md`

## Global Constraints

- Work on the existing `feat/rtp-llm-rollout` feature checkout.
- Opt-in standalone RTP-LLM, FSDP/FSDP2 full-weight export only.
- Preserve the disabled path and replica quarantine/pin semantics.
- No training-worker calls after snapshot export; no live parameter aliases.
- At most latest and in-flight snapshots retained by the publisher.
- Use `.venv-test` (managed by uv) for CPU tests; do not claim GPU validation.

## Task 1: Snapshot Export And Consumption

Files: `verl/checkpoint_engine/weight_snapshot.py`,
`verl/workers/engine_workers.py`, and
`tests/checkpoint_engine/test_weight_snapshot_on_cpu.py`.

Interface: frozen `WeightSnapshot(version, buckets, tensor_count, total_bytes)`;
`export_weight_snapshot(weights, version, bucket_size, is_master)` returns a
manifest on rank zero and None elsewhere. `iter_snapshot_weights(snapshot)`
asynchronously yields CPU named tensors from object-store buckets.

- [x] Add failing tests: mutate model tensors after export and verify snapshot
  values remain unchanged; force object-store failure and verify remaining
  collective generator entries are consumed; verify bucket round-trip.
- [x] Implement immutable CPU copies, bounded buckets, manifest validation,
  nested references, and generator draining without swallowing export errors.
- [x] Add registered collective worker RPC `export_weight_snapshot(global_steps)`.
- [x] Run snapshot tests and inspect worker integration.

## Task 2: Independent Scheduling

Files: `verl/checkpoint_engine/weight_scheduler.py` and
`tests/checkpoint_engine/test_weight_scheduler_on_cpu.py`.

Interface: `WeightPublicationScheduler(manager, on_commit=None)` exposes async
`publish(snapshot)`, `wait_for_version(version, timeout_s)`, `get_status()`,
`pin_replica(rank, reason)`, `unpin_replica(rank)`, `recover_replica(rank)`, and
`close(drain=True)`. Manager exposes existing replica state/pin methods plus
`update_replica_from_snapshot(snapshot, rank, recovery=False)` returning the
existing manager result shape. `on_commit(version, result)` is async and runs
once after the first successful replica commit at that version.

- [x] Add event-controlled failing tests showing publish returns while a
  replica update is blocked and intermediate versions are coalesced.
- [x] Implement monotonic admission, single-update scheduling, pin/unpin,
  quarantine/recovery, bounded manifests, status, wait, and shutdown semantics.
- [x] Test fatal errors, all-unavailable states, and cancellation cleanup.

## Task 3: Replica Adapter And Trainer Integration

Files: `verl/checkpoint_engine/weight_publication.py`,
`verl/checkpoint_engine/base.py`, rollout config Python/YAML files,
fully async trainer/main, and focused integration tests.

- [x] Add failing tests for a snapshot-driven update with no trainer calls and
  for opt-in compatibility validation.
- [x] Add `CheckpointEngineWorker.update_weights_from_snapshot(snapshot)` and
  `SnapshotCheckpointEngineManager.update_replica_from_snapshot(...)`.
- [x] Construct a CPU Ray actor wrapping scheduler/manager and notify rollouter
  only after a successful version commit.
- [x] Export and publish at training update boundaries, with startup/validation
  waits, error propagation, and shutdown cleanup.
- [x] Test asynchronous trainer return, bootstrap, validation, and cleanup.

## Task 4: Verification And Documentation

- [x] Run focused new tests and existing fine-grained update/migration tests.
- [x] Review cross-component ownership, admission races, failure paths, and
  object-reference lifetimes; fix demonstrated gaps.
- [x] Document configuration, memory costs, supported modes, and GPU acceptance
  procedure; run lint, formatting, and diff checks.

## Verification Record

The final combined CPU suite passed 245 tests and 30 subtests, including a real
local Ray cross-process test. That test blocks replica installation, mutates
the training actor's parameters, then verifies the independent publisher can
finish while the training actor is busy and both replicas receive the original
snapshot values. Existing Ray deprecation and unavailable optional-engine
warnings remain. Ruff lint and format checks passed for all 14 changed Python
files, and `git diff --check` passed.

The actual Hydra fully-async configuration composes and converts with
`weight_publication.enabled=True`, `fine_grained_weight_update.enabled=True`,
FSDP2, standalone RTP-LLM, and `hybrid_engine=False`. The default hybrid setting
is correctly rejected; the README example explicitly disables it.

Independent review found invalid recovery of a SERVING replica could poison
the scheduler. Recovery now validates QUARANTINED before entering transfer
failure handling, with regressions for serving and pinned-serving replicas;
genuine recovery transfer failures remain fatal. Scoped re-review found no
remaining issue. Export cancellation settles every rank, and normal shutdown
lets the health watcher finish before the main runner terminates the actor.

No GPU/FSDP distributed execution, large-model memory measurement, or network
throughput benchmark ran on this CPU development host. Changes remain in the
existing feature checkout; no commit or push was performed.
