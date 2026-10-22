# DeepScaleR Data with RTP-LLM

This recipe trains the current model on
[agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset).
It reuses the existing fully asynchronous FSDP2 + RTP-LLM trainer, TP replicas,
weight publication and trajectory scheduler. It is a dataset integration, not
a reproduction of the original DeepScaleR model or staged training recipe.
No RTP native rebuild or new Python dependency is needed for this integration.

Run these commands from the VERL repository with its training environment active.

## Prepare Data

The upstream dataset has only a `train` split, with `problem`, `answer` and
`solution` fields. The preprocessor pins revision
`4500022f6cb0a1456ea5eabe2ff366d4610ee339` by default:

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
python3 -m examples.data_preprocess.deepscaler \
  --local_save_dir "$HOME/verl/data/deepscaler" \
  --test_size 500 --seed 42
```

For an offline cluster, download `deepscaler.json` beforehand and use the same
converter locally. Raw JSONL and Parquet files are also supported:

```bash
python3 -m examples.data_preprocess.deepscaler \
  --local_dataset_path /mnt/fuse/datasets/deepscaler/deepscaler.json \
  --local_save_dir "$HOME/verl/data/deepscaler" \
  --test_size 500 --seed 42
```

Outputs are `train.parquet`, `test.parquet` and `manifest.json`. The manifest
records the source, upstream revision (only for Hub input), seed, split sizes
and filtering counts. Keep these artifacts with the run. Rerunning into the
same directory replaces them, so prepare a new directory when changing the
source, seed or split size during an experiment.

- Only `problem` and an instruction to put the final answer in `\boxed{}` enter
  the user prompt. Neither the reference solution nor answer enters the prompt.
- The unboxed `answer` becomes `reward_model.ground_truth`; the dataset source
  identifier is retained for reward routing.
- Empty questions/answers are dropped and counted. Duplicate questions,
  ignoring whitespace, are removed before the seeded holdout split. Conflicting
  answers for the same question fail preprocessing with the original row indices.
- `extra_info.index` retains the original source row index. Rollout group UIDs
  continue to be assigned by the existing trainer.

The holdout is an internal validation set, not an independent benchmark. Set
`TEST_FILE` or `data.val_files=...` to evaluate on a separately prepared dataset.
If using an external benchmark, check its overlap with the training corpus.

## Train

The wrapper supplies DeepScaleR data/output names and a 2048-token prompt
limit, then delegates to `run_qwen3_5_27b_fsdp2.sh`. `MODEL_PATH`, `TRAIN_FILE`,
`TEST_FILE`, `DATA_DIR`, `N_RESPONSES`, `MAX_RESPONSE_LENGTH`, `LOG_DIR` and
`CKPTS_DIR` remain configurable. All trailing Hydra overrides take precedence.
The model, trainer and scheduler are the same as the existing recipe.

For the existing four trainer GPUs plus three TP4 rollout replicas:

```bash
MODEL_PATH=/mnt/fuse/Qwen/Qwen3.8-27B \
DATA_DIR="$HOME/verl/data/deepscaler" \
N_RESPONSES=4 TRAIN_MINI_BATCH_SIZE=8 \
MAX_RESPONSE_LENGTH=8192 CONCURRENT_SAMPLES_PER_REPLICA=4 \
bash examples/grpo_trainer/run_qwen3_5_27b_fsdp2_deepscaler.sh \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 \
  rollout.nnodes=3 rollout.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.trajectory_migration.enabled=False
```

This example uses the existing NCCL weight-sync backend with KV migration
disabled. Without resource overrides, the parent script uses one node with
four trainer GPUs and four TP1 rollout replicas. Choose context length and
concurrency for the actual GPU memory budget; these are not verified L20C
capacity settings. New dataset runs use separate default checkpoint/log names
so they do not resume a previous MATH-500 experiment accidentally.

The same entry point accepts the GPU snapshot publication overlay:

```text
actor_rollout_ref.rollout.weight_publication.enabled=True
actor_rollout_ref.rollout.weight_publication.transport=nccl_gpu
```

Keep the hardware-validated memory budgets and scheduling overrides from the
[GPU snapshot guide](../../docs/checkpoint_engine/gpu_snapshot_pool.md).
For remote-prefix migration, retain the matching KVCM/Mooncake deployment and
backend overrides from the [validation guide](../../docs/checkpoint_engine/gpu_snapshot_validation.md).
The wrapper also retains the parent script's migration default when no override
is supplied; preparing DeepScaleR data does not provision a KV service.
OSS model, dataset and output paths remain independent of KV storage.

## Reward and Acceptance

Both `agentica-org/DeepScaleR-Preview-Dataset` and `deepscaler` route to the
existing `math_reward` scorer: normalized comparison of the complete last
boxed answer, with 1 for correct and 0 otherwise. This avoids the short answer
tail used by the `math`/DAPO source route. The scorer uses string normalization,
not a general symbolic equivalence prover, and does not reproduce DeepScaleR's
original verifier. The existing custom reward hook can select another verifier.

The wrapper disables the DAPO overlength penalty by default so reward variance
reflects answer correctness. To opt into it, pass
`reward.reward_kwargs.overlong_buffer_cfg.enable=True` and set a buffer length
no greater than `MAX_RESPONSE_LENGTH`.

Use at least two responses per prompt for GRPO. The dataset alone cannot
guarantee mixed rewards or learning in every batch. For a short acceptance run,
append the existing strict checks:

```text
++trainer.learning_acceptance=True
++actor_rollout_ref.actor.optim.verify_parameter_updates=True
++actor_rollout_ref.rollout.verify_weight_updates=True
actor_rollout_ref.actor.optim.lr_warmup_steps=0
actor_rollout_ref.actor.optim.weight_decay=0
trainer.resume_mode=disable
```

Inspect within-group reward variance, effective advantages, finite nonzero
gradients, optimizer parameter deltas and installed rollout weights. A constant
reward batch correctly fails strict acceptance; do not synthesize reward noise
to make it pass. These checks establish an update, not convergence.

CPU regression tests cover offline conversion, split isolation, reward routing
and composition of the real launcher with TP4/GPU-publication overrides:

```bash
python3 -m pytest tests/utils/dataset/test_deepscaler_on_cpu.py -q
```
