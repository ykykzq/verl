# Joint Replica Scheduling Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the
> independent integration tasks and review. Preserve existing uncommitted fixes.

**Goal:** Implement publication-driven replica evacuation, bounded deferral and
capacity-safe trajectory migration with asynchronous tail staleness probes.

**Architecture:** The router owns persistent logical trajectory state and atomic
placement decisions. The independent weight publisher requests and polls updates;
the client reports checkpoints and performs the existing migration handshake.

**Tech Stack:** Python 3.10+, asyncio, Ray, Hydra, RTP-LLM, pytest.

**Spec:** `docs/superpowers/specs/2026-10-19-joint-replica-scheduling-design.md`

## Global Constraints

Opt-in, unchanged legacy behavior, no extra GPU weight snapshot, no trajectory
pruning, no commits/pushes, retain existing fixes. All new RPCs are version fenced.

## Task 1: Router Controller

- [x] Add failing tests in `tests/workers/rollout/test_joint_scheduler_on_cpu.py`
  for new admission closure versus continuation, capacity, packing, deadlines,
  cooldown and version fences; run with `.venv-test/bin/python -m pytest`.
- [x] Implement `verl/workers/rollout/joint_scheduler.py` and integrate the public
  interfaces from the spec into `router.py`. Capacity reservations survive
  checkpoint release and become target ownership only at migration commit.
- [x] Test overlapping publications, unavailable targets, failed migration,
  overload relief, sparse in-place update, and stale probe results.

## Task 2: Client Lifecycle

- [x] Add failing client tests for register/report/finish and cancellation.
- [x] Update `llm_server.py`: read joint config, forward it to the router,
  register around the complete resume loop, report actual progress/prefix IDs,
  finish in `finally`, retain per-segment behavior versions.
- [x] Run existing migration and partial-resume tests as well as new tests.

## Task 3: Tail Scoring

- [x] Add failing backend tests for `score_trajectory_tail` using a fake engine
  stream with actual tensors, including wrong version, abort and malformed loss.
- [x] Implement the RPC in `rtp_llm_async_server.py` using `calculate_loss=2`;
  register under the admission lock, disable cache reuse, check loss alignment,
  slice the tail and return negative cross-entropies with admitted version.
- [x] Add compatible rollout configuration fingerprint to migration capabilities.
- [x] Run RTP-LLM adapter CPU tests.

## Task 4: Publication And Configuration

- [x] Add failing config/scheduler tests for requirements and router selection.
- [x] Add `JointSchedulingConfig` under `WeightPublicationConfig`, YAML defaults
  and export. Wire it through trainer creation and WeightPublisher.
- [x] Prepare all replicas in `publish`; in background use router selection,
  map IDs to replica ranks and poll while deferred. Preserve legacy scheduler,
  commands, cancellation cleanup and latest-version coalescing.
- [x] Run publication/config CPU tests.

## Task 5: Integration And Documentation

- [x] Review the complete patch independently and resolve relevant findings.
- [x] Run focused combined tests and Ruff check/format; document real results.
- [x] Document example capacity/cooldown settings, actual transport and KL-proxy
  limits in the fully-async Chinese README. Report unavailable GPU validation.

## Verification Results (2026-10-19)

- Final combined CPU/Ray regression: 411 passed, 49 subtests passed in 52.86s,
  using `.venv-test/bin/python -m pytest` on Python 3.12. The suite covers weight
  snapshots/publication, router lifecycle, trajectory migration, client
  cancellation, the RTP-LLM adapter, configuration and fully async integration.
- Ray tests ran outside the process sandbox so local worker processes could
  start. Six warnings concern existing optional engines and a Ray API deprecation.
- Ruff check and format verification passed for all 18 touched implementation
  and regression files; `git diff --check` passed.
- Review fixes include atomic evacuation claims, retained probe reservations,
  exact generation-RPC settlement before cancellation cleanup, same-version
  failure recovery, bounded token protection, effective runtime batch limits,
  accepted-ticket cleanup and required partial rollout continuation.
- GPU/KVCM execution, model-specific teacher forcing, memory and throughput
  validation remain deployment acceptance work. No RDMA transport was added.
