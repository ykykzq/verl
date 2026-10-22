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
`4500022f6cb0a1456ea5eabe2ff366d4610ee339` by default. Start with the default,
strict `error` policy to audit unresolved duplicate-answer conflicts:

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
python3 -m examples.data_preprocess.deepscaler \
  --local_save_dir "$HOME/verl/data/deepscaler-audit" \
  --test_size 500 --seed 42 --conflict_policy error
```

For an offline cluster, download `deepscaler.json` beforehand and use the same
converter locally. Raw JSONL and Parquet files are also supported:

```bash
python3 -m examples.data_preprocess.deepscaler \
  --local_dataset_path /mnt/fuse/datasets/deepscaler/deepscaler.json \
  --local_save_dir "$HOME/verl/data/deepscaler-audit" \
  --test_size 500 --seed 42 --conflict_policy error
```

The pinned official dataset contains conflicting answers. Strict preprocessing
scans all rows, writes `audit.json` and `manifest.json`, then exits nonzero
without producing Parquet files. Inspect the audit before selecting a policy.
To explicitly quarantine every row of each unresolved question group, run:

```bash
python3 -m examples.data_preprocess.deepscaler \
  --local_dataset_path /mnt/fuse/datasets/deepscaler/deepscaler.json \
  --local_save_dir "$HOME/verl/data/deepscaler" \
  --test_size 500 --seed 42 --conflict_policy drop
```

A successful run also writes `train.parquet` and `test.parquet`. Every run
requires a fresh output directory: existing managed outputs are never overwritten
or mixed with a new audit. Preparing or auditing data does not start training.

The manifest records the policy, normalization version, status, input SHA-256,
seed, split sizes, empty rows, equivalent duplicate rows and conflict/quarantine
counts. `revision` is set only for Hub input; it remains `null` for local files.
The separate `verified_upstream` field identifies the pinned revision only when
the input bytes match its known SHA-256. Keep both reports with the run.

- Only `problem` and an instruction to put the final answer in `\boxed{}` enter
  the user prompt. Neither the reference solution nor answer enters the prompt.
- The normalized, unboxed `answer` becomes `reward_model.ground_truth`; the
  dataset source identifier is retained for reward routing.
- Empty questions/answers are dropped and counted. Duplicate questions,
  ignoring whitespace, are grouped before any retention or holdout decisions.
  All answers in a retained group must have the same normalized representation.
  Keys are sorted before the seeded shuffle, so input row order does not change
  which questions enter each split.
- `audit.json` records every unresolved group and equivalent duplicate group,
  with original zero-based row indices, raw answers and normalized answers.
  An unresolved conflict is not necessarily a proven mathematical contradiction;
  the normalizer deliberately cannot prove every equivalence.
- `extra_info.index` identifies the retained representative. `source_indices`
  and `source_answers` preserve all equivalent source rows and original labels.
  Reference solutions are not copied. Rollout group UIDs continue to be assigned
  by the existing trainer.

For a successful run, counts obey:
`input_rows = dropped_empty_problem + dropped_empty_answer +
equivalent_duplicate_rows + quarantined_rows + train_rows + test_rows`.
`duplicate_problems` is retained as an alias for `equivalent_duplicate_rows`;
duplicates inside quarantined groups are counted only under `quarantined_rows`.

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
MODEL_PATH=/mnt/fuse/Qwen/Qwen3.5-27B \
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
dedicated `verl.utils.reward_score.deepscaler` scorer. Preprocessing and scoring
share the `deepscaler-format-v1` normalization contract:

- Remove math delimiters (`$...$`, `$$...$$`, `\(...\)` or `\[...\]`)
  and `\boxed{...}` wrappers only when they enclose the whole answer.
  A currency prefix such as `$87.50` is preserved literally.
- Parse LaTeX with the existing `pylatexenc` dependency, canonicalizing
  `\frac`, `\dfrac`, `\tfrac` and `\sqrt` argument braces. For example,
  `$\frac{24}5$` and `\frac{24}{5}` have identical representations.
- Compare whole integers, finite decimals and supported rational fractions
  exactly, without float tolerances. `4.8` and `\frac{48}{10}` become
  `\frac{24}{5}`; `4.8000000001` remains different.
- Preserve unknown syntax, text, units, commas and internal whitespace. There
  is no general symbolic equivalence proof or blanket character removal.

Only the complete last `\boxed{...}` answer is scored, with 1 for a nonempty
normalized match and 0 otherwise. Missing, empty or malformed final boxes score
0, even if an earlier box was correct. Answers are not truncated to a short tail.
Other data sources retain their existing reward routes. This does not reproduce
DeepScaleR's original verifier; the custom reward hook can select another verifier.

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

### Pinned Corpus Validation

The complete 40,315-row JSON at the pinned revision was processed offline with
both policies, `test_size=500`, and `seed=42`. Its SHA-256 is
`4e9e7b18248d982b4498a54a6bdb37b1bb653cf5f77bc309fa67a6c2533c8ba4`.
Strict mode wrote the complete audit and exited 1 without Parquet outputs.
Explicit `drop` mode exited 0 with these counts:

| Category | Count |
| --- | ---: |
| Empty answers | 6 |
| Equivalent duplicate rows removed | 822 |
| Unresolved duplicate question groups | 169 |
| Rows quarantined from those groups | 565 |
| Training questions | 38,422 |
| Validation questions | 500 |

Every original row was accounted for; normalized question sets are disjoint,
normalization is idempotent over the full corpus, and retained labels and source
indices agree with the original data. The tetrahedron rows 538/636 are retained
with the same normalized answer and reward; the cylinder group containing
56/6906 is quarantined in full. These are zero-based source indices.

This checks conversion and conflict handling, not every label's correctness.
Five retained source labels still require manual review: 3153 and 4949 have
incomplete braces; 6303, 6313 and 6368 mix a boxed answer with surrounding
expressions or punctuation. Wrapping their entire normalized label in a new
box does not yield a positive self-match under the final-box scorer. They are
preserved literally, not guessed or silently dropped: `drop` handles duplicate
question conflicts only. Review these records or use an explicitly chosen
verifier before treating the corpus as fully label-validated. No GPU training
or learning acceptance run was performed as part of this preprocessing repair.
