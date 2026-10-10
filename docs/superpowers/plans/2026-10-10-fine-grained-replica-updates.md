# Fine-Grained Replica Updates Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans with independent,
> disjoint-file tasks delegated through superpowers:dispatching-parallel-agents.
> Run each behavioral test red before implementing it, then green and regressions.

**Goal:** Update individual RTP rollout replicas while others serve, and migrate
token trajectories across versions using forced prefill.

**Architecture:** The router owns lifecycle, pins, and published versions. The
checkpoint manager serializes temporary trainer/one-replica process groups and
quarantines failures. Migration has explicit remote_prefix and recompute modes;
the RTP actor validates admission/version twice and disables both caches for
recompute requests.

**Tech Stack:** Python, asyncio, Ray, pytest, OmegaConf, RTP-LLM IPC.

**Spec:** `docs/superpowers/specs/2026-10-10-fine-grained-replica-updates-design.md`

## Global Constraints

- Both features default off; explicitly enable them in the Qwen3.5 27B recipe.
- No second weight copy or rollback; recovery resends the complete weight stream.
- Weight transport is serial; other SERVING replicas continue generation.
- Only SERVING replicas receive requests or become migration targets.
- Pins prevent updates, not serving. The router publishes versions atomically.
- Cross-version migration transfers tokens only and disables local/remote caches.
- Preserve legacy all-replica updates and same-version KVCM behavior.
- Work on the specified clean feature branch, preserving the existing design commit.
- Use uv for the CPU test environment. Commit as ykyzq <yelv.lh@alibaba-inc.com>
  with AI attribution. Do not push. Do not modify RTP core without evidence.

## Task 1: Router Lifecycle And Configuration (Complete)

Files: `verl/workers/rollout/router.py`, `verl/workers/config/rollout.py`,
`verl/workers/config/__init__.py`, rollout configuration YAML as required,
`examples/grpo_trainer/run_qwen3_5_27b_fsdp2.sh`,
`tests/workers/rollout/test_router_lifecycle_on_cpu.py`, config CPU tests.

Interfaces (synchronous atomic router operations):
`pin_replica(server_id, weight_version=None, reason='')`,
`unpin_replica(server_id)`, `begin_replica_update(server_id, weight_version)`,
`commit_replica_update(server_id, weight_version, metadata=None)`,
`fail_replica_update(server_id, error)`,
`begin_replica_recovery(server_id, weight_version)`, `get_replica_states()`.
All state operations return detached dictionaries; states are keyed by server ID.
Begin rejects pins and invalid lifecycle transitions. Metadata refresh may not
overwrite an update's published version. Router decisions preserve scheduler mode.

- [x] Write pin/version/transition, route exclusion (including deterministic),
  reservation invalidation, stale metadata, and config-validation tests.
  Example: begin an update on s0, assert all acquisitions select s1 while s0's
  handle and committed version remain registered; fail then explicitly recover.
- [x] Run `pytest tests/workers/rollout/test_router_lifecycle_on_cpu.py -q` and
  observe missing lifecycle behavior before implementation.
- [x] Add lifecycle state and atomic operations; filter serving routes/targets and
  invalidate sticky/reservations on isolation. Add default-off typed configuration,
  nonnegative retry validation, config normalization, and recipe switches.
- [x] Rerun the new tests and existing router/scheduler/config tests.

## Task 2: Scheduler And Token-Only Client Migration (Complete)

Files: `verl/workers/rollout/trajectory_scheduler.py`,
`verl/workers/rollout/llm_server.py`,
`tests/workers/rollout/test_trajectory_scheduler_on_cpu.py`,
`tests/workers/rollout/test_llm_server_trajectory_migration_on_cpu.py`.

Interface: scheduler diagnostics contain `mode`, `source_version`,
`target_version`. Recompute tickets contain `backend='recompute'`,
`mode='recompute'`, model_id, request_id, prefix_tokens, prefix_digest,
source_version, target_version. Prefix digest uses the existing RTP algorithm.
The client transfers the existing trajectory_state (prompt, generated IDs,
checkpoint index, sampling params) and commits routing only after acceptance.

- [x] Add failing tests: disabled cross-version rejects; enabled cross-version
  chooses recompute without KVCM capability; same-version remains remote_prefix;
  recompute never calls source.prepare and records both versions in history.
- [x] Run both focused migration test files and record the failures.
- [x] Implement explicit modes with model/version safety gates and preserved
  custom gates; construct recompute tickets locally; preserve remote-prefix path.
  Capture target acceptance across enqueue so updates cannot silently discard it.
- [x] Run both files green and verify cancellation and rejected-ticket recovery.

## Task 3: RTP Admission, Forced Prefill, And Recovery (Complete)

Files: `verl/workers/rollout/rtp_llm_rollout/rtp_llm_async_server.py`,
`tests/workers/rollout/rollout_rtp_llm/test_async_server_kv_cache.py`.

Interface: accept both tickets above and existing remote-prefix tickets. Store
accepted mode/version with prefix identity. Immediately before enqueue, require
the accepted target version still matches the actor; otherwise return aborted.
Recompute GenerateConfig sets reuse_cache=False, enable_remote_cache=False.
`abort_weight_update_from_ipc(round_id=None)` may clean the current open round
when the manager lacks its sender-owned ID, keeping admission closed.

- [x] Write failing tests for model/prefix/request/version/admission validation,
  cache flags, enqueue race, abort accepted-ticket cleanup, and full retry after
  a partial weight failure.
- [x] Run the RTP CPU test file and confirm behavioral failures.
- [x] Implement recompute validation and per-request flags without changing
  same-version cache performance. Clear accepted tickets on abort; make recovery
  clean the failed round and allow a subsequent complete begin/finish.
- [x] Run RTP KV and IPC regression tests; inspect core cache flags read-only.

## Task 4: Single-Replica Checkpoint Orchestration And Wiring (Complete)

Files: `verl/checkpoint_engine/base.py`,
`verl/experimental/fully_async_policy/fully_async_trainer.py`,
`verl/experimental/fully_async_policy/fully_async_rollouter.py`,
`tests/checkpoint_engine/test_fine_grained_updates_on_cpu.py`.

Manager constructor adds optional `fine_grained_config` and
`load_balancer_handle`. New APIs match the spec. Map ranks via replica_rank and
server_address. Public manager states are keyed by rank. Preserve legacy return
metrics; enabled results include per-replica status/version/attempts and counts.

- [x] Add CPU fake transport tests around the real router and manager: serial
  topology membership, other replicas serving, pin skips, retries, quarantine,
  fail-fast, recovery, cancellation, finalize/prepare/resume failures and no-router
  fail-closed. Example: fail s0's first stream, verify two full sends and two
  finalizations before s1 starts; no call aborts s1 during s0's update.
- [x] Run the new file red before adding production methods.
- [x] Add async transfer lock and one-replica helper with prepare/init/send/finalize
  under guaranteed cleanup. Await all dispatched participants on errors. Abort
  the IPC round and keep generation closed before retry/quarantine. Resume and
  fetch capabilities before router commit. Omitted ranks select all, including
  pinned status reports; explicit recover alone reopens quarantined replicas.
- [x] Expose rollouter.get_load_balancer() and pass its handle/config at fully
  async checkpoint initialization. Ensure initial capabilities populate versions
  when fine-grained updates are enabled without trajectory migration.
- [x] Run manager and global-step tests green; attempt the GPU server-adapter
  test and record the unavailable hardware/runtime prerequisites below.

## Task 5: Integration Review, Verification, And Commit (Complete)

- [x] Review cross-component races: capability refresh during update, pin versus
  begin, update between accept/commit/enqueue, and cleanup after any prepared
  participant fails. Add a failing regression before each required correction.
- [x] Run the five requested test files plus new manager/router/config tests,
  relevant rollout/IPC/checkpoint CPU suites, lint, and `git diff --check`.
- [x] Record CPU results and the remaining GPU validation requirement. Verify
  both repository statuses and check for unrelated changes.
- [x] Commit implementation on VERL with the requested author and attribution;
  report commit hash, tests, and limits. Leave RTP core untouched if unnecessary.

## Integration Evidence And Decisions

- Task 1: router/configuration red failures observed before code; 72 combined
  lifecycle/router/scheduler/config tests passed in its final focused run.
- Task 2: 11 initial scheduler/client failures preceded implementation. Regression
  tests then exposed admission races, same-replica recompute, mandatory-replan
  thresholds, legacy continuation compatibility, empty-pool handling, metric
  schema mismatch, and backward version extrema. Final focused integration:
  120 passed plus 27 subtests, including real mixed-sample `DataProto.concat`.
- Task 3: 26 initial RTP test failures preceded implementation. Further red/green
  cycles covered cleared acceptance, request-cache environment override, admitted
  version provenance, forced-prefill counts, and collective stream draining.
  Final actor/IPC focused run: 47 passed plus 30 subtests.
- Task 4: 17 initial failures observed, then 17 passed. Three cancellation and
  cleanup regressions failed before fixes, then 20 passed. Trainer wiring failed
  before integration; the combined manager/global-step suite passed 24 tests.
- Integration review exposed NCCL's default persistent communicator. Added
  `CheckpointEngine.prepare_temporary()` as an opt-in backend contract, implemented
  for NCCL; unsupported backends fail at manager initialization. This avoids
  claiming temporary-group safety for transports whose lifecycle was not audited.
  Legacy `prepare()` remains unchanged. Fine-grained calls require an explicit
  non-null `global_steps` because publishing unidentified weights is unsafe.
- Five temporary-communicator/partial-prepare/metadata-socket tests failed before
  fixes; the combined manager/NCCL suite then passed 27 tests. CPU tests exercise
  real NCCL lifecycle code with allocation and collective transport substituted.
- Resolved the original six integration findings with regressions: cached NCCL
  membership, collective iterator draining after IPC failure, safe source fallback
  after rejected migration, lifecycle-aware empty-pool waits, stable batch
  telemetry schema, and version extrema for backward migrations. Tool-agent
  aggregation adds seven regressions, all observed failing before implementation,
  for cross-turn extrema, token provenance, and accumulated migration telemetry.
- Final independent review found an additional stream-draining hole when the
  rollout adapter's dtype conversion fails. Three tests failed before fixing it;
  all four adapter tests now pass, including the real IPC sender, raw stream
  draining, nonleader behavior, and preservation of the original error.
- Bulk generation resume now holds the update lock and only resumes SERVING
  replicas when fine-grained updates are enabled. Four regressions failed before
  this correction, covering transfer, KV resume, and capability failures plus an
  in-progress update. The manager/NCCL/global-step suite then passed 33 tests.
- Final combined CPU regression: **284 passed, 30 subtests passed**, in 29.12s.
  The eight warnings concern optional unavailable engines, Ray API deprecation,
  and existing pytest dataclass collection. Ruff lint and formatting passed on all
  22 changed Python files; recipe shell syntax and `git diff --check` passed.
  Independent review found no remaining blocking issues in the final source.
- Requested GPU test invocation:
  `.venv-test/bin/python -m pytest tests/checkpoint_engine/test_special_server_adapter.py -q --tb=short`.
  It failed during fixture setup with `KeyError: 'ROLLOUT_NAME'`; the fixture also
  requires a CUDA worker pool and `~/models/Qwen/Qwen3-VL-2B-Instruct`. Hardware
  probe reported Darwin, `cuda_available=False`, and `cuda_device_count=0`.
- Actual GPU/NCCL transfer and full-prefill performance require the GB200 runtime;
  this macOS CPU environment cannot provide that acceptance evidence.
- RTP hybrid/dynamic-rebalance paths are outside this change's validated scope.
  No RTP-LLM core modifications were necessary. Its worktree remains clean at
  `b47a25de07`. VERL retains the requested `feat/rtp-llm-rollout` branch and design
  commit, with implementation authored by `ykyzq <yelv.lh@alibaba-inc.com>` and
  AI attribution. No push is performed.

### Reproduce The Final CPU Run

```bash
PATH="$PWD/.venv-test/bin:$PATH" PYTHONPATH="$PWD" .venv-test/bin/python -m pytest \
  tests/checkpoint_engine/test_fine_grained_updates_on_cpu.py \
  tests/checkpoint_engine/test_temporary_nccl_group_on_cpu.py \
  tests/checkpoint_engine/test_global_steps_on_cpu.py \
  tests/checkpoint_engine/test_fsdp_lora_only_checkpoint_on_cpu.py \
  tests/checkpoint_engine/test_sglang_fusion_bucketing_on_cpu.py \
  tests/checkpoint_engine/test_block_placement.py \
  tests/checkpoint_engine/test_sharded_delta.py \
  tests/workers/rollout/test_trajectory_scheduler_on_cpu.py \
  tests/workers/rollout/test_llm_server_trajectory_migration_on_cpu.py \
  tests/workers/rollout/test_router_on_cpu.py \
  tests/workers/rollout/test_router_lifecycle_on_cpu.py \
  tests/workers/rollout/test_llm_server_response_length_cap_on_cpu.py \
  tests/workers/rollout/test_llm_server_routed_experts_on_cpu.py \
  tests/workers/rollout/rollout_rtp_llm/test_async_server_kv_cache.py \
  tests/workers/rollout/rollout_rtp_llm/test_ray_ipc_weight_transfer.py \
  tests/workers/rollout/rollout_rtp_llm/test_rollout_weight_adapter.py \
  tests/workers/config/test_replica_updates_config_on_cpu.py \
  tests/utils/test_config_on_cpu.py \
  tests/experimental/agent_loop/test_tool_agent_versions_on_cpu.py \
  tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py \
  tests/experimental/agent_loop/test_call_tool_on_cpu.py \
  tests/experimental/agent_loop/test_tool_call_id_on_cpu.py -q --tb=short
```
