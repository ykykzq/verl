# RTP-LLM Tensor Parallel Replicas

## Scope

Support a single-node TP=4 rollout replica (and other positive TP sizes fitting
one node), retaining TP=1 behavior. Each replica uses DP=PP=EP=1. Multiple
replicas, CPU weight publication/RDMA relay, independent replica updates,
same-layout KVCM migration and cross-version recompute remain supported.
Cross-node TP and replica-internal DP/PP/EP are rejected explicitly.

## Architecture

Create one RTP engine Ray actor for each checkpoint worker/GPU. All ranks see
the same ordered replica-local CUDA device list; rank r selects local device r.
RTP's native TCPStore/NCCL bootstrap and model loaders supply tensor parallelism.
All ranks start concurrently using a shared replica port block. Only rank zero
owns the generation visitor and is exposed to the router. Partial launch failure
terminates all engine actors belonging to the replica.

Each checkpoint worker sends full named tensor buckets to its corresponding
engine rank through local IPC. The existing RTP receptor loader performs the
rank-specific split. This keeps temporary memory bucketed and does not retain an
additional complete GPU snapshot. Weight updates stop the whole replica, drain
and clear every rank, install every shard, then check identical committed versions
and no unfinished/failed update before reopening the rank-zero admission gate.

Migration flush waits for all ranks. Cache namespaces include layout identity so
TP=1 and TP=4 cannot collide in a shared KVCM domain. Existing layout fingerprints
continue to reject incompatible migrations. Generation and scoring reject direct
calls to non-serving ranks. A failed rank fails the replica as a whole.

## Validation

CPU tests cover topology/port/GPU mapping, boot argument propagation, worker to
rank weight routing, concurrent startup and cleanup, all-rank resume fencing,
migration flush failures, layout namespaces and TP=1 regressions. An opt-in
four-GPU smoke test exercises real engine boot, generation, weight installation
and post-update generation. Hardware execution requires a CUDA RTP environment
and a compatible model; local CPU results do not establish GPU/NCCL correctness.
