# CPU RDMA Weight Relay Implementation Plan

> Use superpowers:subagent-driven-development for the transport and independent
> review; use test-driven development for each contract. Do not commit or push.

**Goal:** Deliver CPU-only, pipelined RDMA weight distribution independent of training.

**Architecture:** A publisher-owned controller maps checkpoint workers to nodes,
starts CPU relays with hard affinity, and distributes immutable snapshot buckets.
The existing scheduler installs node-local manifests with the existing transaction.

**Tech Stack:** Python 3.10+, Ray, PyTorch CPU tensors, Mooncake TransferEngine.

**Spec:** `docs/superpowers/specs/2026-10-20-rdma-weight-relay-design.md`

## Tasks

- [x] Implement registered CPU bucket transport and relay actor in
  `verl/checkpoint_engine/rdma_relay.py`, with CPU contract tests.
  Actor APIs: `load_bucket(bucket) -> descriptor`,
  `pull_bucket(descriptor) -> {descriptor, ref}`, `release_bucket(bucket_id)`,
  `get_status()`, `close()`. Descriptors include node/session/buffer identity,
  tensor layout, byte length and checksum. All reads settle before deregistration.
- [x] Implement `relay_distribution.py`: discover worker node IDs, start one
  relay per node, pipeline chains with bounded concurrency and independent node
  completion, retain active/latest versions and close after consumers settle.
- [x] Add opt-in configuration, trainer source-node placement, worker locality
  RPC and node-local installation; hook staging before replica drain in
  `weight_publication.py` and background staging into scheduler publication.
- [x] Add controller, publisher, worker and configuration regression tests plus
  an opt-in multi-node RDMA smoke test.
- [x] Run independent review, address findings, run existing publication/router
  regressions and Ruff, and document enablement, memory costs and acceptance limits.

## Verification (2026-10-20)

- Combined publication, relay, worker, configuration, routing, RTP-LLM and fully
  async regression: 466 passed, 49 subtests passed, 1 hardware test skipped in
  70.79s with `.venv-test/bin/python -m pytest` on Python 3.12.
- The Ray tests ran outside the process sandbox and used real separate actors,
  CPU tensors and nested object refs. Their test-only socket TransferEngine
  exercises memory ownership and bytes, not actual RDMA hardware.
- Independent review findings resolved: retry earlier predecessors when an
  upstream relay dies after receiving a bucket; keep dead-actor cleanup local
  while treating uncertain live deregistration as fatal; rebuild failed
  same-version distribution for retry/recovery. Scoped re-review found no new
  blocking defects. Ready-node status excludes subsequently failed nodes.
- Ruff check and format verification passed for all 16 touched Python files;
  `git diff --check` passed. Six warnings are existing optional engines and Ray
  deprecation notices.
- Real multi-node Mooncake RDMA equality/cleanup acceptance is implemented in
  `tests/checkpoint_engine/test_rdma_relay_cluster.py` but was not executed on
  this host. GPU installation, bandwidth and training-overlap benchmarks remain
  deployment validation. No commits or pushes were made.
