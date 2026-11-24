# TP-aware GPU Snapshot Pool

`actor_rollout_ref.rollout.weight_publication.transport=nccl_gpu` selects an
independent GPU snapshot publisher for standalone RTP-LLM with FSDP/FSDP2.
It preserves the existing admission/drain/re-prefill scheduling and installs
exactly one rollout replica at a time. CPU relay publication remains available
as `mooncake_rdma`; the original Ray snapshot transport remains the default.

For conversion on trainer GPUs with cached RTP rules and role-specific memory
budgets, see [trainer-side conversion](trainer_conversion.md). The data flow
below describes the default rollout conversion path.

## Data Flow

1. Reserve an immutable version slot before exporting. The trainer's existing
   per-tensor FSDP collective produces full HF tensors on GPU. Rank zero sends
   the required tensors to the conversion workers, one per TP column.
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

All modes retain FSDP's full per-tensor export; they do not directly reshard each
trainer's local FSDP shard. The default per-tensor mode shares a transfer lock
between publication and installation. Bucket mode uses a separate communicator
and lock per TP column for installation, so fetching a sealed older snapshot
can overlap publication of a new one. Version pins protect the fetched storage.
Slot or working-buffer exhaustion applies bounded backpressure. Serving on
other replicas continues, but GPU compute, memory bandwidth and NCCL traffic
can still contend with decoding.

## Bucket Pipeline

Both options below are under `actor_rollout_ref.rollout.weight_publication.gpu_snapshot`:

| `bucket_bytes` | `pipeline_depth` | Behavior |
| --- | --- | --- |
| 0 | 0 | Original per-tensor publication |
| Positive | 0 | Packed source buckets and independent per-TP installation fetch |
| Positive | 2 | Two input buffers, two output batches, and receive/convert/distribute overlap |

For example, `bucket_bytes=536870912` and `pipeline_depth=2` enable a 512 MiB
target bucket size. Only depths 0 and 2 are accepted; depth 2 requires
`2 * bucket_bytes <= max_tensor_bytes`. These flags belong in the resolved
launch configuration on every node. Changing modes requires fresh actors;
communicator topology cannot be switched in place.

The pipeline has separate input and output credits. Receiving bucket `k+1`
can overlap converting `k` and distributing `k-1`. An input credit is returned
only after conversion has completed and released its IPC input reference.
An output credit is returned after persistent shard copies and remote DMA
complete and RTP acknowledges that output batch's release. Input broadcast,
shard distribution and per-TP fetch use independent communicators. Enqueuing a
CUDA operation does not acknowledge completion. Sealing drains every stage,
performs the converter's final flush and validates full runtime tensor coverage.

Converters are chosen independently per TP column: prefer actor rank zero's
node, then other trainer nodes, then stable worker order. This reuses the
fully-async NCCL multi-sender topology; trainer ranks on the source node act as
broadcast relays. Every trainer still drains the same FSDP generator. Each
input broadcast completes on device before its participants advance to the
next FSDP collective, preserving collective ordering across communicators.
The overlap is on the rollout pipeline; FSDP all-gather and input broadcast do
not run concurrently. Replica installation remains limited to one at a time.

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
| `bucket_bytes` | 0 | Target packed source/install bucket size; 0 disables buckets |
| `pipeline_depth` | 0 | 0 disables pipelining; 2 enables two input and two output slots |
| `operation_timeout_s` | 120 s | Native transfer and temporary-buffer wait bounds |
| `reserve_timeout_s` | 600 s | Version-slot backpressure bound |

Before KV-cache allocation, RTP reserves persistent storage, conversion working
memory and temporary tensor storage plus its IPC bucket. Pipeline mode doubles
the conversion-memory reservation to cover two live output batches. Version
slots and pipeline buffer slots are distinct: persistent versions are bounded
by `slots` and `max_bytes_per_gpu`, while all input/install temporary buffers
on a worker share `max_tensor_bytes`. Layout planning checks actual per-owner
storage across all version slots.

A single HF or runtime tensor may exceed `bucket_bytes`; that tensor is put
in its own bucket and must still fit `max_tensor_bytes`. Large HF tensors need
full temporary storage on conversion ranks; runtime distribution is chunked.
When two large inputs cannot coexist within the budget, allocation waits for
the preceding conversion to release its input. The source likewise drops an
older, completed broadcast buffer before allocating a larger one. A configured
limit must fit the largest source tensor and each local runtime tensor.
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
install timings. Pipeline metrics additionally report the selected topology,
input/output high-water counts and per-bucket `pipeline_events`. These events
measure host spans including RPC and waiting; they are not kernel utilization
or precise DMA bandwidth measurements. Strict learning and remote-KV acceptance
remain separate from weight transfer success.

Shutdown closes admission before draining pending allocation, reception,
conversion, distribution and installation. Partially allocated inputs or
failed distribution poison the pool and retain owned storage for worker
restart; closing a failed pool cannot discard those owners.

Run the [validation guide](gpu_snapshot_validation.md) in this order: directed
NCCL and storage fidelity, actual RTP TP4 conversion/install, native remote KV
and strict cross-replica reuse, then RL with nonzero gradients/parameter deltas.
Local CPU tests do not establish hardware bandwidth, memory fit or convergence.
