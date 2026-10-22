# GPU Snapshot Validation

The `nccl_gpu` publication path needs separate transport, model-installation,
and end-to-end RL evidence. The opt-in test below checks CUDA storage and NCCL
using synthetic runtime tensors. It does not instantiate RTP, convert a real
checkpoint, run tensor-parallel inference, or certify a 27B model's memory fit.

## Transport and Storage

Run from the same VERL checkout installed on every Ray node. Each node needs
compatible CUDA PyTorch, CuPy with NCCL bindings, and working NCCL connectivity.
Use idle GPUs: the test reserves one full GPU per actor and terminates its own
actors when finished or when a transfer fails. Models, checkpoints, and logs
remain on OSS; no model download is needed for this test.

```bash
export VERL_ROOT=/home/admin/rl-validation/verl
export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_ADDRESS=10.64.186.151:27383
export VERL_RUN_GPU_SNAPSHOT_TEST=1
export VERL_GPU_SNAPSHOT_REQUIRE_MULTINODE=1
export VERL_GPU_SNAPSHOT_TEST_TIMEOUT_S=180
export PYTORCH_ALLOC_CONF=expandable_segments:False
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_IB_GID_INDEX=3
python -m pytest tests/checkpoint_engine/test_gpu_snapshot_nccl_on_gpu.py -q -s -rs
```

Replace the addresses, NIC, and GID for the actual cluster. The harness forwards
NCCL settings, allocator settings, `PYTHONPATH`, and `LD_LIBRARY_PATH` to its
actors. It sends tensor payload through `DirectedNCCL`; Ray carries identities,
layout/chunk metadata, versions, NCCL setup, and verification results only.
Small GPU-to-CPU copies are made solely for test checksums, not publication.

The directed-transfer case uses two GPU actors and checks both directions for
`int64`, `float16`, `bfloat16`, `float32`, and `float64`. Comparison is byte-exact,
including arbitrary floating-point bit patterns, and the producer uses a
nondefault CUDA stream. Results include verified byte counts and SHA-256 hashes.

The storage cases use two replicas with synthetic TP sizes one and two, needing
two and four GPUs respectively. They use the production layout planner and
`GPUSnapshotWorker` chunk storage/reconstruction. Every owner stays in the
tensor's TP column; uneven chunk tails and all reconstructed bytes are checked.
Version 7 stays pinned while version 8 is published; version 9 is rejected while
both slots are protected. After unpinning version 7, its slot is reused for
version 9, stale pinning fails, and both versions 8 and 9 remain byte-correct.
Placement prefers different nodes for different replicas, and the multinode
flag requires at least two actual node identities in the results.

Without `VERL_RUN_GPU_SNAPSHOT_TEST=1`, all three test cases explicitly skip
before importing CUDA/Ray dependencies. The TP2 case skips if fewer than four
Ray GPUs are advertised. An enabled run with no GPU node or a requested
multinode run on only one GPU node fails. A skip is not hardware evidence.
These tests do not exercise `GPUSnapshotPool` source ingestion, RTP conversion,
or installation; those remain part of the full acceptance run below.

Healthy shutdown first drains the publication scheduler, input ingestion, and
outstanding snapshot reads. It then closes the trainer source and all storage
communicators concurrently, since NCCL teardown can wait for its peers. A fresh
publisher can initialize those workers after this healthy close. Failed or
uncertain transfers retain their owners and reject reuse until the affected
actors restart. Include healthy close/reconfiguration in hardware acceptance;
CPU lifecycle tests cannot establish native communicator teardown behavior.

## Two-Node L20C Acceptance

The original handoff used two eight-GPU L20C hosts, four trainer GPUs and three
TP4 rollout replicas, with `/mnt/fuse/Qwen/Qwen3.8-27B`. Its initial publication
and GPU KV byte tests passed, but optimizer steps completed were zero. The new
GPU publication path must establish its own results on this model/topology.

First rebuild RTP with the handoff build fixes and record the installed
`libth_transformer` path and `get_kvcm_build_info()` on both hosts. Keep the
matching Mooncake master/store and storage-group configuration. The KVCM
metadata RPC endpoint is `10.64.186.151:6381`, Mooncake master RPC is `50151`,
and the master's HTTP metrics listener is `9004`:

```bash
export MASTER_METRICS=9004
export KVCM_MOONCAKE_MASTER=10.64.186.151:50151
export MC_GID_INDEX=3
python -m rtp_llm.utils.kvcm_probe --service-only \
  --address 10.64.186.151:6381 --instance-group l20c_rl_pinned_20261021 \
  --min-free-bytes 67108864
RTP_RUN_NATIVE_KV_TEST=1 \
KVCM_SERVER_ADDRESS=10.64.186.151:6381 \
KVCM_INSTANCE_GROUP=l20c_rl_pinned_20261021 \
python -m pytest "${RTP_LLM_HOME}/rtp_llm/test/smoke/test_native_kv_probe_cluster.py" -q -s
```

For the test command, `RTP_LLM_HOME` is the RTP source checkout containing the
smoke tests; run the Python environment containing its matching rebuilt wheel.
The KVCM server process itself also needs `MASTER_METRICS=9004` before startup.
Changing the training driver's environment does not change an existing server.
The CPU preflight checks actual `/health` and `/metrics` capacity. It cannot
prove RDMA transfer or reserve storage; retain the separate native GPU test.

The existing 27B launcher accepts Hydra overrides. The following acceptance
overlay retains its launcher and selects the original resource topology. Keep
the original environment's Ray authentication, CUDA library paths and matching
RTP wheel setup. Use the real unique-prefix data fixtures, and place output
directories on the OSS mount:

```bash
export MODEL_PATH=/mnt/fuse/Qwen/Qwen3.8-27B
export DATA_DIR=/home/admin/rl-validation/validation-l20c-20261021-6e92e76e/data-unique-prefix
export LOG_DIR=/mnt/fuse/oss/yelv.lh/validation/gpu-snapshot/logs
export CKPTS_DIR=/mnt/fuse/oss/yelv.lh/validation/gpu-snapshot/checkpoints
export N_RESPONSES=4 TRAIN_MINI_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=1280 MAX_RESPONSE_LENGTH=512
export TOTAL_ROLLOUT_STEPS=16 CONCURRENT_SAMPLES_PER_REPLICA=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.95
export WEIGHT_PUBLICATION_WAIT_TIMEOUT_S=1800

bash examples/grpo_trainer/run_qwen3_5_27b_fsdp2.sh \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 \
  rollout.nnodes=3 rollout.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.weight_publication.enabled=True \
  actor_rollout_ref.rollout.weight_publication.transport=nccl_gpu \
  actor_rollout_ref.rollout.weight_publication.joint_scheduling.enabled=False \
  actor_rollout_ref.rollout.trajectory_migration.enabled=True \
  actor_rollout_ref.rollout.trajectory_migration.checkpoint_tokens=32 \
  actor_rollout_ref.rollout.trajectory_migration.min_score_improvement=-1.0 \
  ++actor_rollout_ref.rollout.trajectory_migration.strict_remote_hit=True \
  +actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_server_address=10.64.186.151:6381 \
  +actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_instance_group=l20c_rl_pinned_20261021 \
  '+actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_model_sdk_config="[{\"type\":\"mooncake\",\"location\":\"*\",\"put_replica_num\":1}]"' \
  +actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_put_timeout_ms=10000 \
  +actor_rollout_ref.rollout.trajectory_migration.backend_options.kvcm_get_timeout_ms=10000 \
  ++trainer.learning_acceptance=True \
  ++actor_rollout_ref.actor.optim.verify_parameter_updates=True \
  ++actor_rollout_ref.rollout.verify_weight_updates=True \
  actor_rollout_ref.actor.optim.weight_decay=0 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  async_training.trigger_parameter_sync_step=1 \
  reward.reward_kwargs.overlong_buffer_cfg.enable=False \
  '+data.apply_chat_template_kwargs.enable_thinking=False' \
  trainer.test_freq=-1 trainer.save_freq=-1 trainer.resume_mode=disable \
  "++ray_kwargs.ray_init.address=${RAY_ADDRESS}" \
  '++ray_kwargs.ray_init.runtime_env.env_vars.MASTER_METRICS="9004"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.KVCM_MOONCAKE_MASTER="10.64.186.151:50151"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.MC_GID_INDEX="3"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_HCA="mlx5_bond_0"' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_GID_INDEX="3"'
```

This overlay uses default GPU limits of two slots, 10 GiB persistent snapshot
storage per GPU, 8 GiB conversion working memory, 4 GiB maximum runtime tensor,
and 256 MiB chunks. These are bounds, not a measured 27B/L20C memory plan. The
`0.95` utilization setting is tentative and has not been verified on hardware;
the old `0.4` setting cannot cover the reserved GPU snapshot/conversion memory
and runtime weights on L20C. Two BF16 27B snapshots distributed over 12 GPUs
require roughly 9.12 GB per GPU before layout-specific overhead; the 10 GiB
persistent limit only clears that theoretical lower bound.
The conservative conversion budget may reject this model; runtime weights, KV
cache, conversion buffers, NCCL buffers and persistent slots must all fit.
Measure the described TP runtime layout and peak allocated/reserved CUDA memory
on the actual wheel, then set `weight_publication.gpu_snapshot.*` limits from
that evidence. Do not claim this command fits L20C based on the synthetic test.

## Required Evidence

Record repository revisions, wheel hash/build info, model identity, physical
node identities, TP ranks, master/store revision, complete launch configuration,
and the per-stage timings/errors. Successful setup or initial version 0 alone
does not satisfy acceptance. Require all of the following:

- A complete initial snapshot and updated snapshots installed on all 12 rollout
  ranks, with identical layout hashes, expected version transitions, and
  byte/digest evidence for source and installed runtime tensors. Include a
  pinned older-version fetch during publication and retirement only after its
  readers finish. Validate the pool coordinator and RTP conversion/install path.
- At least one real optimizer step with multi-response UID groups, nonzero
  within-group reward variance, nonzero valid-token advantages, finite nonzero
  gradient norm and a nonzero parameter delta. Keep weight decay and warmup at
  zero for this check. Retain raw group sizes/rewards, truncation/EOS information,
  and bounded scoring evidence if strict learning acceptance fails.
- Rollout requests after that step use the newly published version and changed
  installed parameters. Initial weight installation is not evidence of a
  training update.
- With `strict_remote_hit=True`, a real trajectory resumes on another replica
  and reports `remote_reuse_len > 0` for the requested prefix, with source and
  destination identities, checkpoint tokens, cache/version information, and
  successful continuation. Metadata location hits, token replay, and the
  standalone native byte probe do not establish trajectory KV reuse.
- Retain failures, timeouts and capacity refusals. NCCL uncertainty requires
  terminating the affected actors; do not free their source/receive buffers and
  retry on the same communicator. SDK soft timeout 100 likewise requires rank
  restart according to RTP's native policy.

No local macOS CUDA/NCCL or real TP4/RL acceptance result is claimed by this
test/document change.

## Local Regression Evidence (2026-10-22)

- Focused VERL publication, scheduling, snapshot, IPC, TP-control, KV-control,
  configuration and learning-acceptance suite: 288 passed, 54 subtests passed.
  Three opt-in CUDA cases skipped because this host has no CUDA hardware.
- Existing real local Ray publication integration: 1 passed. This validates
  coordinator independence and snapshot ownership, not GPU communication.
- RTP runtime-conversion dependency-isolated CPU harness: 16 passed, including
  actual TP split/fusion functions. Probe configuration and loopback Mooncake
  service preflight suite: 20 passed.
- VERL scoped Ruff/format checks, both repositories' diff checks and launcher
  shell syntax pass. RTP new-code lint passes with the existing root-logger
  warning excluded; unrelated baseline weight-manager lint is unchanged.
- The supplied RTP build patch reverse-check passes, confirming that the
  handoff patch is applied. Native compilation and installed-wheel capability
  checks were not run on this macOS host.

Rebuild/install the matching RTP and VERL code on every cluster node before
running the hardware steps above.
