# KV Data Path and Acceptance Repair Plan

**Goal:** Make native cache failures diagnosable, reject false-positive migration and learning acceptance, and bound RDMA staging for oversized tensors.

**Architecture:** Keep the existing KVCM client and immutable publication boundaries. Add structured native completion status, native GPU-buffer probes, strict acceptance checks, and bounded relay transfer chunks. Preserve existing behavior outside opt-in acceptance checks.

**Constraints:** Python 3.10 compatibility; TP=4; no token-only fallback may count as remote KV migration. No SDK/master pair may be labelled verified without native cross-node byte-equality evidence. Local development has no CUDA/RDMA; SSH to the validation cluster currently times out.

- [x] Native status: add success, timeout and failed-write tests; propagate SDK codes across TP RPC and Python bindings. Native tests require the Linux build below.
- [x] Native data probe: use the linked SDK with registered GPU buffers for allocation/write/read and byte comparison; expose isolated startup and cross-node validation entry points, preserving failure diagnostics.
- [x] Dependency provenance: pin the master to the public KVCM dependency reference, verify its archive hash and compare the exact SDK's four-argument PutStart symbol with the incompatible five-argument deployed master. Report unresolved binary source commits separately; native compatibility remains unverified.
- [x] Migration: expose remote reuse tokens, attempts, failures and fallback reasons; add strict cross-replica remote-hit validation and focused tests.
- [x] RDMA: test oversized tensors and concurrent budget limits; transfer bounded chunks and reconstruct complete weights; record phase durations.
- [x] Learning: reject zero group variance, masked/zero advantages, nonfinite/zero gradients and unchanged parameters; add opt-in optimizer evidence and acceptance tests.
- [x] Build/exit: fix capability test runfiles and add isolated-process native import/exit regression.
- [x] Verify CPU suites and static checks; document native/GPU checks that could not run. No commit/push is requested in this turn.
- [ ] Cluster acceptance: Linux CUDA/Bazel build and shutdown tests, native cross-node byte equality, two TP4 replicas with strict remote hits, nonzero RL parameter updates, then RDMA plus KV migration. These require the inaccessible validation cluster.

## Local Evidence

- Relevant VERL CPU suite: 331 passed, 1 skipped, 67 subtests passed.
- Four real-Ray integration cases passed, including chunk metadata across relays, a failed middle relay, trainer-independent publication and cancelled migration cleanup. The no-relay publication regression found by this run was fixed and its test rerun successfully.
- Native probe configuration/subprocess contract: 4 unittest cases passed. Launcher overrides parsed with Hydra; Python lint and both worktrees' whitespace checks passed.
- NativeKVProbe passed C++20 syntax checking against the downloaded SDK and local Torch headers. This does not substitute for the Linux Bazel/CUDA build or native runtime tests.
- Independent read-only review found no additional actionable defects. Exact RPM source provenance and real GPU protocol compatibility remain unverified.
