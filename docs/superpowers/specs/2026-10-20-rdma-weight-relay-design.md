# CPU RDMA Weight Relay

The user requested the RDMA relay extension to the existing independent weight
publisher and joint trajectory scheduler. Preserve the existing uncommitted work.

## Contract

- Opt in with `weight_publication.transport=mooncake_rdma`; default `ray` remains.
- Trainer exports immutable CPU snapshots as before. Publication only admits a
  manifest; CPU packing and RDMA distribution run in the publisher background.
- One CPU relay actor per participating rollout node, plus a master relay on
  the source export node. Hard node affinity prevents accidental remote installs.
- Each bucket is packed into registered CPU memory with typed tensor metadata
  and a checksum. Receivers RDMA-read from the preceding relay, publish an
  immutable local Ray bucket and become the next source. Multiple buckets flow
  through the chain concurrently with bounded staging memory.
- Control RPCs carry descriptors and nested refs, never inter-relay tensor data.
  No GPU relay buffers, full GPU snapshot copies or training collectives are added.
- Preserve the named-tensor format and existing RTP-LLM TP/PP partitioning during
  installation; this change does not invent a second sharding implementation.
- Source buffers remain registered until dependent reads settle, including
  cancellation and failures. Checksum/version/schema errors fail closed.
- A broken relay does not strand downstream nodes: subsequent nodes may read
  the last healthy predecessor. Node failures are reported independently.
- A replica installs only a complete manifest on each of its worker nodes.
  Snapshot transport completes before draining that replica; installed versions
  still use the existing commit/quarantine protocol.
- Distribution retains at most an active and latest publication. Superseded
  versions release local refs after consumers finish. Explicit close waits for
  transfers before releasing actors and registered storage.
- Missing Mooncake/RDMA capability is an actionable error; no silent TCP or Ray
  data-plane fallback in RDMA mode. Hardware validation must be reported honestly.

## Validation

CPU tests use the real pack/unpack and chain controller with a byte-copy fake
TransferEngine. Cover mixed dtypes, noncontiguous/empty/scalar tensors, corrupt
descriptors, failed reads/registration, cancellation, bounded staging, pipeline
progress, failed intermediate nodes, local installation and legacy compatibility.
An opt-in cluster smoke test uses actual Mooncake RDMA actors and verifies tensor
equality and registered-memory cleanup on two or more nodes.
