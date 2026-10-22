# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Prepare DeepScaleR math problems for VERL GRPO, with a disjoint holdout."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download

from verl.utils.reward_score.deepscaler import NORMALIZATION_VERSION, normalize_answer

DATA_SOURCE = "agentica-org/DeepScaleR-Preview-Dataset"
DATA_REVISION = "4500022f6cb0a1456ea5eabe2ff366d4610ee339"
DATA_SHA256 = "4e9e7b18248d982b4498a54a6bdb37b1bb653cf5f77bc309fa67a6c2533c8ba4"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


class ConflictingAnswersError(ValueError):
    def __init__(self, stats, audit):
        super().__init__(
            f"DeepScaleR has conflicting answers in {stats['conflict_groups']} question groups; "
            "inspect audit.json or explicitly select --conflict_policy drop to quarantine entire groups"
        )
        self.stats = stats
        self.audit = audit


def prepare_splits(dataset, test_size=500, seed=42, conflict_policy="error"):
    if conflict_policy not in ("error", "drop"):
        raise ValueError("conflict_policy must be error or drop")
    missing = {"problem", "answer"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"DeepScaleR raw data is missing columns: {sorted(missing)}")

    stats = {
        "input_rows": len(dataset),
        "dropped_empty_problem": 0,
        "dropped_empty_answer": 0,
        "duplicate_problems": 0,
        "equivalent_duplicate_rows": 0,
        "conflict_groups": 0,
        "conflict_rows": 0,
        "quarantined_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
    }
    audit = {
        "normalization_version": NORMALIZATION_VERSION,
        "conflicts": [],
        "equivalent_duplicates": [],
        "empty_rows": [],
    }
    groups = {}
    for index, example in enumerate(dataset):
        problem, answer = example["problem"], example["answer"]
        for field, value in (("problem", problem), ("answer", answer)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"DeepScaleR row {index}: {field} must be a string or null")
        raw_answer = answer
        problem, answer = (problem or "").strip(), normalize_answer(answer or "")
        if not problem:
            stats["dropped_empty_problem"] += 1
            audit["empty_rows"].append({"index": index, "reason": "empty_problem", "raw_answer": raw_answer})
            continue
        if not answer:
            stats["dropped_empty_answer"] += 1
            audit["empty_rows"].append({"index": index, "reason": "empty_answer", "raw_answer": raw_answer})
            continue
        key = " ".join(problem.split())
        group = groups.setdefault(key, {"problem": problem, "rows": []})
        group["rows"].append({"index": index, "raw_answer": raw_answer, "normalized_answer": answer})

    # Classify complete groups before retaining a representative or assigning splits.
    # Sorted keys make question membership independent of raw input order.
    examples = []
    for key in sorted(groups):
        group = groups[key]
        rows = group["rows"]
        entry = {"problem": key, "rows": rows}
        if len({row["normalized_answer"] for row in rows}) != 1:
            audit["conflicts"].append(entry)
            stats["conflict_groups"] += 1
            stats["conflict_rows"] += len(rows)
            continue
        if len(rows) > 1:
            audit["equivalent_duplicates"].append(entry)
            stats["equivalent_duplicate_rows"] += len(rows) - 1
        examples.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": f"{group['problem']}\n\n{INSTRUCTION}"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": rows[0]["normalized_answer"]},
                "extra_info": {
                    "index": rows[0]["index"],
                    "source_indices": [row["index"] for row in rows],
                    "source_answers": [row["raw_answer"] for row in rows],
                },
            }
        )
    stats["duplicate_problems"] = stats["equivalent_duplicate_rows"]
    stats["unique_problems"] = len(groups)
    stats["eligible_problem_groups"] = len(examples)
    if stats["conflict_groups"] and conflict_policy == "error":
        raise ConflictingAnswersError(stats, audit)
    stats["quarantined_rows"] = stats["conflict_rows"]
    if not 0 < test_size < len(examples):
        raise ValueError(f"test_size must be between 1 and {len(examples) - 1} after filtering and deduplication")

    random.Random(seed).shuffle(examples)
    splits = {}
    for split, rows in (("train", examples[test_size:]), ("test", examples[:test_size])):
        for row in rows:
            row["extra_info"]["split"] = split
        splits[split] = datasets.Dataset.from_list(rows)
        stats[f"{split}_rows"] = len(rows)
    return datasets.DatasetDict(splits), stats, audit


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dataset_path", help="Raw JSON, JSONL, Parquet, or a directory with deepscaler.json.")
    parser.add_argument("--revision", default=DATA_REVISION, help="Hugging Face revision; ignored for local input.")
    parser.add_argument("--local_save_dir", "--local_dir", default="~/verl/data/deepscaler")
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face dataset cache directory.")
    parser.add_argument("--test_size", type=int, default=500, help="Number of unique problems reserved for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conflict_policy",
        choices=("error", "drop"),
        default="error",
        help="error: audit all unresolved conflicts and fail; drop: quarantine entire conflicting question groups.",
    )
    args = parser.parse_args()

    destination = Path(args.local_save_dir).expanduser()
    for name in ("train.parquet", "test.parquet", "manifest.json", "audit.json"):
        if (destination / name).exists():
            raise FileExistsError(f"Refusing to overwrite {destination / name}; use a fresh output directory")
    local_path = None
    if args.local_dataset_path:
        local_path = Path(args.local_dataset_path).expanduser()
        if local_path.is_dir():
            local_path = local_path / "deepscaler.json"
        if not local_path.is_file():
            raise FileNotFoundError(f"DeepScaleR input not found: {local_path}")
        input_path = local_path
    else:
        input_path = Path(
            hf_hub_download(
                DATA_SOURCE,
                "deepscaler.json",
                repo_type="dataset",
                revision=args.revision,
                cache_dir=args.cache_dir,
            )
        )
    loaders = {".json": "json", ".jsonl": "json", ".parquet": "parquet"}
    if input_path.suffix.lower() not in loaders:
        raise ValueError("DeepScaleR input must be raw JSON, JSONL, or Parquet")
    input_sha256 = file_sha256(input_path)
    dataset = datasets.load_dataset(
        loaders[input_path.suffix.lower()],
        data_files=str(input_path),
        split="train",
        cache_dir=args.cache_dir,
    )

    conflict_error = None
    try:
        splits, stats, audit = prepare_splits(dataset, args.test_size, args.seed, args.conflict_policy)
    except ConflictingAnswersError as error:
        conflict_error = error
        splits, stats, audit = {}, error.stats, error.audit
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for split, dataset in splits.items():
        dataset.to_parquet(str(destination / f"{split}.parquet"))
    manifest = {
        "data_source": DATA_SOURCE,
        "status": "conflicts" if conflict_error else "ready",
        "conflict_policy": args.conflict_policy,
        "normalization_version": NORMALIZATION_VERSION,
        "input_sha256": input_sha256,
        "verified_upstream": (
            {"data_source": DATA_SOURCE, "revision": DATA_REVISION, "sha256": DATA_SHA256}
            if input_sha256 == DATA_SHA256
            else None
        ),
        "revision": None if local_path else args.revision,
        "local_dataset_path": str(local_path.resolve()) if local_path else None,
        "seed": args.seed,
        "test_size": args.test_size,
        "instruction": INSTRUCTION,
        **stats,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if conflict_error:
        raise SystemExit(str(conflict_error))


if __name__ == "__main__":
    main()
