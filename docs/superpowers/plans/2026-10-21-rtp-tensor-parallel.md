# RTP-LLM Tensor Parallel Implementation Plan

**Goal:** Support a single-node TP=4 replica throughout generation and independent
weight updates while preserving trajectory scheduling.

**Architecture:** One engine actor per GPU, one serving rank per replica, native
RTP tensor splitting, replica-wide admission and version checks.

**Tech Stack:** Python, Ray, PyTorch CUDA IPC, RTP-LLM TCPStore/NCCL and KVCM.

**Spec:** `docs/superpowers/specs/2026-10-21-rtp-tensor-parallel-design.md`

## Constraints

- TP ranks must fit one node with distinct GPU IDs. DP=PP=EP=1.
- Existing TP=1 public actor names and request contract remain compatible.
- All rank updates must succeed before generation resumes.
- CPU tests must not require an installed RTP binary or CUDA.
- Do not claim real TP=4 hardware validation without executing it.

## Tasks

- [x] Add failing topology, boot configuration and weight-routing tests.
- [x] Launch one actor per rank with shared rendezvous and correct CUDA context;
      validate configuration and clean up partial launches.
- [x] Route each worker's bucket stream to its own rank; fence resume on all
      committed versions, and flush all ranks for migration.
- [x] Run focused tests, add failure/cancellation and TP=1 regression coverage.
- [x] Add opt-in four-GPU smoke test and configuration/documentation.
- [x] Review the complete integration and run affected regression suites.

## Verification

2026-10-21: the combined publication, relay, router, joint scheduling, RTP and
fully-async regression run passed 485 tests and 61 subtests; two opt-in hardware
tests were skipped. Ray tests required local socket binding outside the sandbox.
Ruff check/format and git diff whitespace checks passed. Independent native RTP
integration review and the final startup-timeout review found no remaining blockers.

An exploratory run of all worker configuration tests additionally encountered
one unrelated model-config test failure because the local
`/Users/liuhang/models/Qwen/Qwen2.5-0.5B` checkpoint is absent. The affected rollout
configuration suites are included in the passing combined run.

Hardware acceptance remains unexecuted: no CUDA/RTP-LLM four-GPU environment or
multi-node RDMA/KVCM setup is available on this host. The GPU test and README
provide the deployment acceptance procedure; CPU tests do not prove hardware
performance or NCCL correctness.
