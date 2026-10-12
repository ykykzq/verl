# Independent Weight Publication

## Objective

Let fully asynchronous RTP-LLM training resume after exporting an immutable
weight snapshot. Replica draining, network transfer, installation, retries,
pinning, and recovery run in a separate Ray actor and never call training
workers after export completes.

## Architecture

Training workers collectively consume the existing full named-tensor export.
For the initial FSDP/FSDP2 implementation, rank zero copies tensors into CPU
buckets and puts them in Ray's object store. Other ranks consume the complete
generator to participate in FSDP collectives. The trainer awaits every export
RPC, then publishes only a manifest containing nested object references.
Snapshots do not alias live model tensors. Export errors must drain generators
where possible and settle all worker calls before training can resume.

A CPU-only Ray publication actor holds the latest complete manifest and one
in-flight manifest. A newer publication replaces work not yet started; an
in-flight replica update finishes with its original snapshot. The next replica
receives the newest available version. This bounds retained publication data
to two snapshots, plus a transient snapshot being exported by training.

Rollout checkpoint workers fetch CPU buckets and pass full named tensors to
the existing RTP-LLM IPC adapter. No trainer joins replica transfer collectives.
This uses additional host memory and Ray object-store/network bandwidth; it
does not promise NCCL/RDMA throughput or durable recovery after actor failure.

## Scheduling And Safety

Reuse CheckpointEngineManager's update lifecycle: close target admission,
abort and drain its requests, clear reusable KV, install the complete snapshot,
verify the version, then publish SERVING. Failure retries use the same immutable
snapshot; final failure quarantines only that replica.

Update one replica at a time. Prefer the oldest serving, unpinned replica.
Skip replicas already at or beyond the latest version. Preserve pin semantics;
unpin triggers catch-up. Recovery explicitly installs the retained latest
snapshot. Versions are nonnegative integers and publications are monotonic.
Background fatal errors are observable through publish, status, and waits.

Reset rollouter's existing staleness budget after the first successful replica
commit for a published version, never merely because export finished. Retain
normal queue backpressure. Startup and explicit validation wait for publication;
steady training only awaits export and lightweight admission. Shutdown drains
or cancels safely, releases snapshot references, and terminates the actor.

## Compatibility

The feature is opt-in, requires fine-grained updates and standalone RTP-LLM,
and initially supports FSDP/FSDP2 full-weight exports. Reject hybrid/dynamic
resource scheduling and nonmerged LoRA configurations. Keep existing NCCL
publication unchanged when disabled. Existing trajectory interruption and
cross-version recompute behavior remains in use.

## Validation

CPU tests cover immutable export, collective draining on failure, real snapshot
consumption, latest-version coalescing, publication while an update is blocked,
pin/unpin, failure isolation, recovery, health propagation, startup/validation
barriers, and shutdown. Existing replica and migration regression suites must
continue to pass. GPU throughput and memory use need a separate cluster run.
