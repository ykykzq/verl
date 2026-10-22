# TP-aware GPU Snapshot Pool

`actor_rollout_ref.rollout.weight_publication.transport=nccl_gpu` selects an
independent GPU snapshot publisher for standalone RTP-LLM with FSDP/FSDP2.
It preserves the existing admission/drain/re-prefill scheduling and installs
exactly one rollout replica at a time. CPU relay publication remains available
as `mooncake_rdma`; the original Ray snapshot transport remains the default.

## Data Flow

1. Reserve an immutable version slot before exporting. The trainer's existing
   per-tensor FSDP collective produces full HF tensors on GPU. Rank zero sends
   the required tensors to the TP workers of one conversion replica.
2. Each RTP worker uses its actual loader, TP rank and resident runtime layout
   to convert those tensors without overwriting inference weights. CUDA IPC
   connects each checkpoint worker to its local RTP process.
3. Stripe the resulting TP-local runtime tensors across replicas in the same
   TP column, using bounded chunks. For three TP4 replicas, TP rank 2's shards
   live only on the three GPUs serving TP rank 2. Payloads stay on GPU; Ray
   carries manifests, layout metadata, IPC handles and acknowledgements.
4. Seal only after all runtime tensors and every shard have arrived. Trainer
   execution can resume after export and publication admission. A separate
   scheduler selects one replica, drains or pauses its requests, gathers only
   its TP-local tensors with directed NCCL sends/receives, and installs them
   directly without another HF-to-runtime transformation.

The first implementation retains FSDP's full per-tensor export and sends source
tensors to the conversion TP ranks that need them. It does not directly
reshard each trainer's local FSDP shard. These source transfers and conversion
are part of export latency. Publication and installation share a transfer lock
at tensor boundaries; slot or working-buffer exhaustion applies bounded
backpressure. Serving on other replicas continues, but GPU compute, memory
bandwidth and NCCL traffic can still contend with decoding.

NCCL removes CPU tensor staging and Python payload serialization from this
path. It does not remove PCIe from machines whose GPU/NIC topology uses PCIe.
See the [NCCL point-to-point contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)
and [CuPy communicator API](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.nccl.NcclCommunicator.html).

## Version and Memory Limits

The default is two version slots. The newest sealed version and any version
pinned by an update cannot be evicted. A pin covers drain, transfer and install,
not just NCCL reads. Superseded unpinned versions can be reclaimed. If all slots
are protected, the next export waits up to `reserve_timeout_s`.

| Option under `weight_publication.gpu_snapshot` | Default | Scope |
| --- | --- | --- |
| `slots` | 2 | Maximum resident/staging versions |
| `max_bytes_per_gpu` | 10 GiB | All persistent slots on one GPU |
| `max_conversion_bytes` | 8 GiB | Pending input clones, outputs and estimated loader workspace |
| `max_tensor_bytes` | 4 GiB | Aggregate temporary source/install tensor storage per GPU |
| `chunk_bytes` | 256 MiB | Runtime shard transfer size |
| `operation_timeout_s` | 120 s | Native transfer and temporary-buffer wait bounds |
| `reserve_timeout_s` | 600 s | Version-slot backpressure bound |

Before KV-cache allocation, RTP reserves the three byte budgets above plus its
IPC bucket. Layout planning also checks the actual per-owner storage across
all slots. Large individual HF tensors still require a full temporary tensor
on conversion ranks; runtime distribution itself is chunked. A configured
limit must fit that largest source tensor as well as each local runtime tensor.
Custom loader kernels may require more than their conservative workspace
estimate; real CUDA peak-memory validation remains necessary. These defaults
are not evidence that Qwen3.8-27B fits a particular L20C memory configuration.

## Failure and Validation

Duplicate tensors, missing tensors, incompatible TP layouts, malformed
manifests and capacity violations reject publication. RTP runtime install
validates exact shape/dtype/key coverage and retains optional installed-weight
comparison through `verify_weight_updates`.

The snapshot pool has unique shards, so loss of a storage worker affects the
whole pool. A native timeout, cancellation with uncertain completion or an
unclassified IPC failure poisons the transport and retains its owners until
worker restart. It must not reuse the communicator or release memory while
DMA might still reference it. This is more conservative than target-only
quarantine for a proven application-level error.

`WeightPublisher.get_status()` exposes per-version slots/readers, per-GPU shard
bytes, staging and tensor progress, fatal state and export/conversion/transfer/
install timings. Strict learning and remote-KV acceptance remain separate from
weight transfer success.

Run the [validation guide](gpu_snapshot_validation.md) in this order: directed
NCCL and storage fidelity, actual RTP TP4 conversion/install, native remote KV
and strict cross-replica reuse, then RL with nonzero gradients/parameter deltas.
Local CPU tests do not establish hardware bandwidth, memory fit or convergence.
