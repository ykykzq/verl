# Trainer-side conversion and version-only scheduling

This integrates the V4 weight path and the V5 policy from the
`rl-ad-version-only-v5-code-results-20261111.zip` handoff. It complements the
[GPU snapshot pool](gpu_snapshot_pool.md); the original rollout converter,
periodic checkpoints and `version_kl` policy remain the defaults.

## Weight path and memory

With `gpu_snapshot.conversion_location=trainer`, startup builds a static
`ConversionPlan` from the actual RTP loader rules and runtime TP layout. It
validates source names, order, shapes, dtypes, FSDP metadata, TP assignments,
conversion rules and derived buffer budgets. All replicas in a TP column must
agree on the plan hash. Only pointer-free rule/metadata bundles cross Ray.

`trainer_tp_ranks=[0,1,2,3]` maps target TP ranks 0–3 to distinct trainer ranks.
Each rank still traverses the full FSDP export in collective order; this is not
direct transfer of its resident FSDP shards. Assigned trainers execute the
cached RTP rules and send final runtime shards directly to rollout storage
with NCCL. Caching avoids repeated rule construction, not tensor conversion.
Each conversion round owns separate tensors, dependency counters and device
state. Installing a snapshot uses the runtime tensors without converting again.

Trainer conversion requires `cache_conversion_plan=true`, `pipeline_depth=2`
and `memory_planning=role`. It never falls back to a rollout converter. A sealed
version requires all mapped trainers and every runtime shard to finish. Slot
pins and the one-replica-at-a-time installation lock remain in effect.
Device completion gates buffer release. Uncertain native transfers retain their
owners and require worker restart. Local conversion initialization failures
still drain the FSDP iterator before reporting failure, so peers can finish
their outstanding collectives.

Role planning reserves snapshot storage, installation and communication buffers
on rollout GPUs, but no trainer-conversion workspace there. A rollout converter
in role mode instead requires an explicit `converter_replica_rank` selected
before KV allocation. TP ranks freeze the replica's KV/sequence capacity to the
minimum safe value across ranks. Trainer free-memory preflight includes the
static conversion peak, largest source tensor and 2 GiB headroom. These checks
do not replace measuring peak memory with the actual wheel and model.

## D recipe

Use the same VERL code and matching RTP Python code/wheel on every node,
including trainer processes that restore the conversion plan. This integration
adds `request_kv_checkpoint` in C++; rebuilding the wheel is necessary when
upgrading from the local pre-V4 RTP baseline. Old wheels fail the selected-stream
capability check before model loading. Keep the existing protocol-matched
Mooncake KVCM service; model/data/artifact paths may remain on OSS.

```bash
export MODEL_PATH=/path/to/Qwen2.5-32B-Instruct
export DATA_DIR=/path/to/preprocessed/deepscaler
export KVCM_SERVER_ADDRESS=kvcm-host:6381
export KVCM_INSTANCE_GROUP=your-mooncake-storage-group
export RAY_ADDRESS=ray-head:port
bash examples/grpo_trainer/run_qwen25_32b_deepscaler_joint.sh \
  "++ray_kwargs.ray_init.address=${RAY_ADDRESS}"
```

Supply the cluster's existing NCCL, KVCM and runtime-library environment. The
launcher does not create services or reproduce machine-specific preload paths.
Its defaults select 4 trainer GPUs + 3 TP4 rollout replicas, 10 accepted updates,
no checkpoint exports and a 500-question greedy evaluation before/after training.
Use a disjoint held-out validation file. It disables the step sample quota,
enables continuous decoding with selected-request pauses, and sets:

| Policy | Value |
| --- | --- |
| `staleness_mode` | `version` (no KL tasks or probe reservations) |
| `staleness_version_gap` | 2, measured against installed SERVING versions |
| `max_defer_s` | 120 seconds, update eligibility fallback |
| `min_decode_tokens` / `min_decode_window_s` | 0 tokens / 2 seconds |
| `update_cooldown_s` / `evacuation_timeout_s` | 5 seconds / 5 seconds |

The launcher's 12 GiB snapshot, 6 GiB temporary tensor and 32 GiB conversion
budgets reproduce the supplied settings; they are not a claim of fit on every
GPU. Override `SNAPSHOT_STORAGE_BYTES`, `SNAPSHOT_TENSOR_BYTES`,
`SNAPSHOT_CONVERSION_BYTES` and `JOINT_KV_CAPACITY_TOKENS` as measured. Remaining
Hydra overrides are appended last. New mode defaults also appear in the four
generated PPO configurations.

Strict learning acceptance checks valid masks, within-group reward variation,
finite/nonzero gradients and parameter changes. The D recipe permits up to
32 audited zero-signal batches to be replaced before an optimizer update;
nonfinite or malformed batches still fail. General defaults allow zero retries.
Rejected data and time are recorded. This changes sample consumption and must
be included in experiment comparisons. Final training requests settle before
evaluation; validation requests bypass trajectory migration.

## Evidence and remaining work

The supplied single-run comparison reports 10 real updates per group, with D
at 431 versus fully-async at 603 effective tokens/s. D improved version lag but
submitted 861 recompute continuations, including 613 on the same replica. Those
counts are logical decisions, not completed GPU prefills. The 120-second
fallback did not trigger; the existing 5-second evacuation rule still caused
updates. Thirteen KV export-preparation timeouts recovered. Neither group's
500-question evaluation change was statistically significant, and the groups
consumed different prompts.

This integration does not add the suggested cost-aware policy separating
evacuation from forced update, or change native FIFO/top-up scheduling. Rerun
native build, plan-hash/byte equivalence, selected-request KV migration and
the two-group RL acceptance for the merged revision. Local CPU tests establish
control-flow and metadata behavior, not distributed CUDA correctness or learning
equivalence.

## Local integration checks (2026-11-11)

- VERL combined CPU regression: 462 tests and 21 subtests passed. Three
  additional real local Ray tests passed for delayed-enqueue cancellation,
  queue backpressure and failing-driver exit.
- RTP isolated conversion/installation tests: 23 passed. Run them directly as
  `python rtp_llm/model_loader/test/test_runtime_snapshot_on_cpu.py`; pytest
  package discovery on a wheel-free host can import native RTP operators.
- The new launcher passed shell syntax, captured-argument Hydra composition
  and production rollout/publication configuration validation without starting
  training. VERL scoped Ruff/format checks and both repositories' diff checks
  passed. RTP conversion module/test lint passed; pre-existing lint findings
  in `weight_manager.py` and `rpc_engine.py` were unchanged.
- Native target
  `//rtp_llm/cpp/engine_base/stream/test:generate_stream_state_test`, wheel build
  and distributed CUDA/RL acceptance were not run on this macOS host. Use the
  cluster's existing CUDA13/KVCM build configuration for those checks.
