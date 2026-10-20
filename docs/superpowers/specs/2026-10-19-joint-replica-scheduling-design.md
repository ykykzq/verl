# Weight-centered replica and trajectory scheduling

The user-authorized design is implemented on the existing RTP-LLM adaptation,
including the outstanding retest fixes in this checkout. No commits or pushes
are part of this request.

## Contract

- Opt in with `weight_publication.joint_scheduling.enabled`; legacy behavior is
  unchanged when disabled. Requires independent publication, fine-grained
  updates, checkpointed migration, cross-version recompute and
  `async_training.partial_rollout=true` to preserve interrupted trajectories.
- Publishing a snapshot closes NEW logical trajectory admission on older
  replicas. Existing trajectories retain their identity across checkpoint gaps.
  Their continuations and compatible incoming migrations remain admissible.
- A router-owned controller chooses low-occupancy replicas for evacuation.
  Checkpoint migration packs work into compatible same-version destinations;
  evacuated sources update first. Deadlines and sparse tails trigger in-place
  updates with explicit recompute tickets for survivors. Newer publications
  coalesce without resetting the oldest outstanding update deadline.
- Replica update cooldown and per-trajectory migration/decode dwell prevent
  repeated interruptions. The elapsed-time dwell is mandatory; the additional
  token floor expires at the deferral deadline so stalled streams cannot prevent
  progress forever. Updating/quarantined replicas are never destinations.
- Capacity is an explicit conservative token budget (`kv_capacity_tokens`),
  rounded to `kv_block_size_tokens`, reserving prompt plus the remaining maximum
  response. `max_batch_size` is a deployment-profiled roofline limit. Both fresh
  admissions and pending/committed migrations consume capacity. Unknown or
  insufficient capacity fails closed; no fabricated GPU telemetry is used.
- Migration uses existing remote-prefix KVCM tickets for equal versions and
  token-only recompute tickets for newer versions. Model/config compatibility,
  version and cache-domain checks remain mandatory. Failed transfers retain the
  source and release the reservation. No moves to older versions.
  An unavailable source may recover by explicit token-only recompute on an equal
  version; this requires joint mode and a distinct recovery ticket flag.
- Version lag gates asynchronous, bounded tail scoring. RTP's teacher-forced
  per-token loss supplies raw model log probabilities on the actual prefix.
  Compare old/new tail scores with the nonnegative sampled k3 KL proxy. This is
  an estimate on sampled tokens, not exact categorical KL; sampling warpers can
  bias it. Failed/stale probes cannot trigger migration. Probe results carry
  source and target versions and scoring never blocks the publication loop.
  Scoring reserves temporary prefix KV and one slot on both replicas until both
  RPCs settle, including after local timeout or logical trajectory completion.
- Preserve behavior versions/logprobs across continuation; re-prefill does not
  rewrite the history. No GRPO group pruning or algorithm change is introduced.
- CPU immutable snapshot publication is reused. A new RDMA relay transport is
  outside this scheduling change; no full GPU snapshot copy is added.

## Interfaces

The router accepts `joint_scheduling_config` and exposes:

```python
register_trajectory(request_id, prompt_tokens, max_new_tokens)
report_trajectory_progress(request_id, server_id, generated_tokens,
                           weight_version, prefix_ids)
finish_trajectory(request_id)
prepare_weight_update(version)
select_replica_for_update(version)  # {server_id, reason} or {server_id: None}
```

Registration precedes acquire; finish runs in `finally`. The router owns
assignment, reservations, cooling windows and update selection. Publisher maps
server IDs to ranks and polls selection without holding up publish RPCs.
Backend RPC `score_trajectory_tail(prefix_ids, tail_tokens,
expected_weight_version)` returns `{weight_version, log_probs}`. It participates
in backend abort/drain and disables cache reuse for teacher forcing.

## Validation

CPU deterministic scheduling tests cover admission, checkpoint gaps, packing,
capacity contention, stale plans, cooldown, deadline fairness, version
coalescing, failure isolation and asynchronous probe lifecycle. Client/backend
contract tests cover cleanup on cancellation, raw-tail score alignment and
version rejection. Publisher tests exercise the actual router-driven choice.
GPU/RDMA performance and engine runtime need the deployment environment.
