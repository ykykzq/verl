"""DeepScaleR preprocessing, reward routing and RTP launcher integration."""

import hashlib
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
    assert manifest["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()

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


def test_existing_outputs_cannot_be_mixed_with_a_failed_or_new_run(tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([{"problem": "A", "answer": "1"}, {"problem": "B", "answer": "2"}]))
    output = tmp_path / "out"
    assert preprocess(source, output, "--test_size", "1").returncode == 0
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    source.write_text(json.dumps([{"problem": "A", "answer": "1"}, {"problem": "A", "answer": "2"}]))
    result = preprocess(source, output, "--test_size", "1")
    assert result.returncode != 0
    assert "fresh output directory" in result.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original


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
        (r"\boxed{$87.50}", "$87.50", 1.0),
        (r"\boxed{337}", r"\boxed{337}", 1.0),
    ],
)
def test_deepscaler_scores_math_answers_without_tail_truncation(source, response, answer, expected):
    assert default_compute_score(source, response, answer) == expected


def real_cases():
    fixture = Path(__file__).parent / "fixtures/deepscaler_conflicts.json"
    return json.loads(fixture.read_text())["rows"]


@pytest.mark.parametrize("source", [SOURCE, "deepscaler"])
@pytest.mark.parametrize("answer", [r"$\frac{24}5$", r"\frac{24}{5}"])
def test_real_fraction_formats_have_identical_rewards(source, answer):
    assert default_compute_score(source, r"\boxed{\frac{24}{5}}", answer) == 1.0
    assert default_compute_score(source, r"\boxed{\frac{23}{5}}", answer) == 0.0
    assert default_compute_score(source, r"\boxed{\frac{24}{5}} then \boxed{7}", answer) == 0.0
    assert default_compute_score(source, r"\frac{24}{5}", answer) == 0.0


def test_equivalent_real_duplicates_share_normalized_truth_and_keep_original_answers(tmp_path):
    source = tmp_path / "real.json"
    rows = real_cases()[:2] + [{"problem": "Other question", "answer": "42"}]
    source.write_text(json.dumps(rows))
    output = tmp_path / "out"
    result = preprocess(source, output, "--test_size", "1")
    assert result.returncode == 0, result.stderr
    prepared = sum((pq.read_table(output / f"{split}.parquet").to_pylist() for split in ("train", "test")), [])
    tetrahedron = next(row for row in prepared if row["prompt"][0]["content"].startswith("Tetrahedron"))
    assert tetrahedron["reward_model"]["ground_truth"] == r"\frac{24}{5}"
    assert tetrahedron["extra_info"]["source_indices"] == [0, 1]
    assert tetrahedron["extra_info"]["source_answers"] == [r"$\frac{24}5$", r"\frac{24}{5}"]
    assert default_compute_score(SOURCE, r"\boxed{\frac{24}5}", tetrahedron["reward_model"]["ground_truth"]) == 1.0
    assert tetrahedron["prompt"][0]["content"] == (
        rows[0]["problem"] + "\n\nLet's think step by step and output the final answer within \\boxed{}."
    )


@pytest.mark.parametrize("policy", ["error", "drop"])
def test_real_conflict_policy_writes_complete_audit_and_never_selects_one_label(tmp_path, policy):
    source = tmp_path / "real.json"
    source.write_text(json.dumps(real_cases() + [{"problem": "Other question", "answer": "42"}]))
    output = tmp_path / "out"
    result = preprocess(source, output, "--test_size", "1", "--conflict_policy", policy)
    assert (result.returncode == 0) == (policy == "drop"), result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    assert manifest["conflict_policy"] == policy
    assert manifest["normalization_version"] == audit["normalization_version"]
    assert manifest["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["conflict_groups"] == 1
    assert manifest["conflict_rows"] == 2
    assert manifest["equivalent_duplicate_rows"] == 1
    assert len(audit["conflicts"]) == 1
    assert audit["conflicts"][0]["rows"] == [
        {"index": 2, "raw_answer": "312", "normalized_answer": "312"},
        {"index": 3, "raw_answer": "179", "normalized_answer": "179"},
    ]
    if policy == "error":
        assert manifest["status"] == "conflicts"
        assert not (output / "train.parquet").exists()
        assert not (output / "test.parquet").exists()
    else:
        assert manifest["status"] == "ready"
        assert manifest["quarantined_rows"] == 2
        assert (manifest["train_rows"], manifest["test_rows"]) == (1, 1)
        prepared = sum((pq.read_table(output / f"{split}.parquet").to_pylist() for split in ("train", "test")), [])
        assert all("cylindrical" not in row["prompt"][0]["content"] for row in prepared)


def test_mixed_duplicate_groups_are_resolved_before_order_independent_splitting():
    from datasets import Dataset

    from examples.data_preprocess.deepscaler import prepare_splits

    rows = [
        {"problem": "A", "answer": r"$\frac12$"},
        {"problem": "A", "answer": "0.5"},
        {"problem": "A", "answer": r"\frac{1}{2}"},
        {"problem": "B", "answer": "312"},
        {"problem": "B", "answer": "179"},
        {"problem": "B", "answer": "$312$"},
        {"problem": "C", "answer": "42"},
        {"problem": "D", "answer": "43"},
        {"problem": None, "answer": "1"},
        {"problem": "Empty", "answer": None},
        {"problem": "Empty math", "answer": "$ $"},
    ]
    previous = None
    for ordered in (rows, list(reversed(rows)), rows):
        splits, stats, audit = prepare_splits(Dataset.from_list(ordered), test_size=1, seed=42, conflict_policy="drop")
        prompts = {
            split: {row["prompt"][0]["content"].split("\n")[0] for row in data} for split, data in splits.items()
        }
        assert prompts["train"].isdisjoint(prompts["test"])
        assert prompts["train"] | prompts["test"] == {"A", "C", "D"}
        if previous is not None:
            assert prompts == previous
        previous = prompts
        assert stats["equivalent_duplicate_rows"] == 2
        assert stats["quarantined_rows"] == 3
        assert stats["dropped_empty_problem"] == 1
        assert stats["dropped_empty_answer"] == 2
        assert stats["train_rows"] + stats["test_rows"] + 2 + 3 + 1 + 2 == len(rows)
        assert len(audit["conflicts"][0]["rows"]) == 3


@pytest.mark.parametrize(
    "answer,expected",
    [
        (r"$\frac{24}5$", r"\frac{24}{5}"),
        (r"\(\dfrac{24}{5}\)", r"\frac{24}{5}"),
        (r"$$\tfrac{48}{10}$$", r"\frac{24}{5}"),
        ("4.8", r"\frac{24}{5}"),
        (r"\sqrt2", r"\sqrt{2}"),
        (r"\sqrt[3]{2}", r"\sqrt[3]{2}"),
        (r"\text{a b}", r"\text{a b}"),
        (r"\[\frac{24}5\]", r"\frac{24}{5}"),
        (r"\frac{1}{0}", r"\frac{1}{0}"),
        (r"\frac{24}{", r"\frac{24}{"),
        (r"2007 + \frac\pi 2", r"2007 + \frac{\pi}{2}"),
        (r"\boxed{337}", "337"),
        (r"$\boxed{337}$", "337"),
        (r"$ \boxed{2, 3, 7} $", "2, 3, 7"),
        ("$87.50", "$87.50"),
        (r"\$24", r"\$24"),
        ("$1$ and $2$", "$1$ and $2$"),
    ],
)
def test_normalization_preserves_math_structure(answer, expected):
    from verl.utils.reward_score.deepscaler import normalize_answer

    assert normalize_answer(answer) == expected
    assert normalize_answer(expected) == expected


@pytest.mark.parametrize(
    "response,answer",
    [
        (r"\boxed{4.8000000001}", "4.8"),
        (r"\boxed{12}", "1,2"),
        (r"\boxed{\text{ab}}", r"\text{a b}"),
        (r"\boxed{1} then \boxed{", "1"),
        (r"\boxed{}", "$ $"),
        (r"\boxed{1} then \boxed 2", "1"),
        (r"\boxed{1} then \boxed", "1"),
        (r"\boxed % last", "1"),
    ],
)
def test_reward_does_not_guess_equivalence_or_use_an_earlier_box(response, answer):
    assert default_compute_score(SOURCE, response, answer) == 0.0


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
