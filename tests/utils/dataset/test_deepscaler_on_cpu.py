"""DeepScaleR preprocessing, reward routing and RTP launcher integration."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir

from verl.utils.reward_score import default_compute_score

REPO = Path(__file__).resolve().parents[3]
SOURCE = "agentica-org/DeepScaleR-Preview-Dataset"


def preprocess(source, output, *options):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.data_preprocess.deepscaler",
            "--local_dataset_path",
            str(source),
            "--local_save_dir",
            str(output),
            "--cache_dir",
            str(output.parent / "hf-cache"),
            *options,
        ],
        cwd=REPO,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("suffix", ["json", "jsonl", "parquet"])
def test_preprocessing_preserves_answers_without_leakage_and_deduplicates_before_split(tmp_path, suffix):
    rows = [{"problem": f"Compute {i} + 1.", "answer": str(i + 1), "solution": "SECRET SOLUTION"} for i in range(6)]
    rows.extend(
        [
            {"problem": " Compute  0 + 1. ", "answer": "1", "solution": "SECRET SOLUTION"},
            {"problem": "No answer", "answer": " ", "solution": "SECRET SOLUTION"},
            {"problem": " ", "answer": "42", "solution": "SECRET SOLUTION"},
        ]
    )
    source = tmp_path / f"raw.{suffix}"
    if suffix == "parquet":
        pq.write_table(pa.Table.from_pylist(rows), source)
    elif suffix == "jsonl":
        source.write_text("\n".join(json.dumps(row) for row in rows))
    else:
        source.write_text(json.dumps(rows))

    output = tmp_path / "prepared"
    result = preprocess(source, output, "--test_size", "2", "--seed", "17")
    assert result.returncode == 0, result.stderr
    train = pq.read_table(output / "train.parquet").to_pylist()
    test = pq.read_table(output / "test.parquet").to_pylist()
    assert (len(train), len(test)) == (4, 2)
    assert {row["extra_info"]["index"] for row in train}.isdisjoint(row["extra_info"]["index"] for row in test)
    for split, samples in (("train", train), ("test", test)):
        for row in samples:
            index = row["extra_info"]["index"]
            assert set(row) == {"prompt", "data_source", "ability", "reward_model", "extra_info"}
            assert row["data_source"] == SOURCE
            assert row["ability"] == "math"
            assert row["reward_model"] == {"style": "rule", "ground_truth": str(index + 1)}
            assert row["extra_info"]["split"] == split
            assert row["prompt"] == [
                {
                    "role": "user",
                    "content": (
                        f"Compute {index} + 1.\n\n"
                        "Let's think step by step and output the final answer within \\boxed{}."
                    ),
                }
            ]
            assert "SECRET SOLUTION" not in json.dumps(row)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["input_rows"] == 9
    assert manifest["dropped_empty_problem"] == 1
    assert manifest["dropped_empty_answer"] == 1
    assert manifest["duplicate_problems"] == 1
    assert manifest["train_rows"] == 4
    assert manifest["test_rows"] == 2
    assert manifest["revision"] is None  # Local data must not claim the upstream commit.

    repeated = tmp_path / "repeated"
    result = preprocess(source, repeated, "--test_size", "2", "--seed", "17")
    assert result.returncode == 0, result.stderr
    assert pq.read_table(repeated / "train.parquet").to_pylist() == train
    assert pq.read_table(repeated / "test.parquet").to_pylist() == test


def test_conflicting_duplicate_answers_are_rejected(tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([{"problem": "Same?", "answer": "1"}, {"problem": "Same?", "answer": "2"}]))
    output = tmp_path / "out"
    result = preprocess(source, output, "--test_size", "1")
    assert result.returncode != 0
    assert "conflicting answers" in result.stderr
    assert not (output / "train.parquet").exists()


@pytest.mark.parametrize("test_size", ["0", "2"])
def test_split_must_leave_nonempty_training_and_validation_sets(tmp_path, test_size):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([{"problem": "A?", "answer": "1"}, {"problem": "B?", "answer": "2"}]))
    result = preprocess(source, tmp_path / "out", "--test_size", test_size)
    assert result.returncode != 0
    assert "test_size" in result.stderr


@pytest.mark.parametrize("source", [SOURCE, "deepscaler"])
@pytest.mark.parametrize(
    "response,answer,expected",
    [
        (r"Reasoning. \boxed{\frac{1}{2}}", "0.5", 1.0),
        (r"Reasoning. \boxed{\sqrt{2}}", r"\sqrt{2}", 1.0),
        (r"\boxed{41}", "42", 0.0),
        ("unfinished reasoning, 42", "42", 0.0),
        (r"\boxed{1} corrected to \boxed{2}", "2", 1.0),
        ("\\boxed{" + "1," * 200 + "2}", "1," * 200 + "2", 1.0),
    ],
)
def test_deepscaler_scores_math_answers_without_tail_truncation(source, response, answer, expected):
    assert default_compute_score(source, response, answer) == expected


def test_deepscaler_launcher_composes_with_tp4_and_gpu_publication_overrides(tmp_path):
    environment = os.environ.copy()
    environment.update(
        RAY_DATA_HOME=str(tmp_path),
        LOG_DIR=str(tmp_path / "logs"),
        CKPTS_DIR=str(tmp_path / "checkpoints"),
        MODEL_PATH="/models/current-qwen",
        N_RESPONSES="4",
        MAX_RESPONSE_LENGTH="2048",
    )
    for key in ("TRAIN_FILE", "TEST_FILE", "DATA_DIR", "PROJECT_NAME", "EXPERIMENT_NAME"):
        environment.pop(key, None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'python3() { printf "%s\\n" "$@"; }; export -f python3; exec bash "$@"',
            "launcher-test",
            str(REPO / "examples/grpo_trainer/run_qwen3_5_27b_fsdp2_deepscaler.sh"),
            "rollout.nnodes=3",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
            "actor_rollout_ref.rollout.weight_publication.enabled=True",
            "actor_rollout_ref.rollout.weight_publication.transport=nccl_gpu",
            "actor_rollout_ref.rollout.trajectory_migration.enabled=False",
            "data.val_files=/data/external-validation.parquet",
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    args = args[args.index("-m") :]
    assert args[:2] == ["-m", "verl.experimental.fully_async_policy.fully_async_main"]
    with initialize_config_dir(config_dir=str(REPO / "verl/experimental/fully_async_policy/config"), version_base=None):
        config = compose(config_name="fully_async_ppo_trainer", overrides=args[2:])
    assert config.data.train_files == str(tmp_path / "data/deepscaler/train.parquet")
    assert config.data.val_files == "/data/external-validation.parquet"
    assert config.actor_rollout_ref.model.path == "/models/current-qwen"
    assert config.actor_rollout_ref.rollout.name == "rtp_llm"
    assert config.actor_rollout_ref.rollout.n == 4
    assert config.actor_rollout_ref.rollout.response_length == 2048
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 4
    assert config.actor_rollout_ref.rollout.weight_publication.transport == "nccl_gpu"
    assert config.actor_rollout_ref.rollout.weight_publication.enabled
    assert not config.actor_rollout_ref.rollout.trajectory_migration.enabled
    assert config.rollout.nnodes == 3
    assert not config.reward.reward_kwargs.overlong_buffer_cfg.enable
    assert "deepscaler" in config.trainer.experiment_name.lower()
