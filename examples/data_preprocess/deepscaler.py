# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Prepare DeepScaleR math problems for VERL GRPO, with a disjoint holdout."""

import argparse
import json
import random
from pathlib import Path

import datasets

DATA_SOURCE = "agentica-org/DeepScaleR-Preview-Dataset"
DATA_REVISION = "4500022f6cb0a1456ea5eabe2ff366d4610ee339"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def prepare_splits(dataset, test_size=500, seed=42):
    missing = {"problem", "answer"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"DeepScaleR raw data is missing columns: {sorted(missing)}")

    stats = {
        "input_rows": len(dataset),
        "dropped_empty_problem": 0,
        "dropped_empty_answer": 0,
        "duplicate_problems": 0,
    }
    examples = []
    seen = {}
    for index, example in enumerate(dataset):
        problem, answer = example["problem"], example["answer"]
        for field, value in (("problem", problem), ("answer", answer)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"DeepScaleR row {index}: {field} must be a string or null")
        problem, answer = (problem or "").strip(), (answer or "").strip()
        if not problem:
            stats["dropped_empty_problem"] += 1
            continue
        if not answer:
            stats["dropped_empty_answer"] += 1
            continue
        # Deduplicate before splitting so repeated questions cannot enter both sets.
        key = " ".join(problem.split())
        if key in seen:
            previous_index, previous_answer = seen[key]
            if answer != previous_answer:
                raise ValueError(f"DeepScaleR rows {previous_index} and {index} have conflicting answers")
            stats["duplicate_problems"] += 1
            continue
        seen[key] = (index, answer)
        examples.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": f"{problem}\n\n{INSTRUCTION}"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"index": index},
            }
        )
    if not 0 < test_size < len(examples):
        raise ValueError(f"test_size must be between 1 and {len(examples) - 1} after filtering and deduplication")

    random.Random(seed).shuffle(examples)
    splits = {}
    for split, rows in (("train", examples[test_size:]), ("test", examples[:test_size])):
        for row in rows:
            row["extra_info"]["split"] = split
        splits[split] = datasets.Dataset.from_list(rows)
        stats[f"{split}_rows"] = len(rows)
    return datasets.DatasetDict(splits), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dataset_path", help="Raw JSON, JSONL, Parquet, or a directory with deepscaler.json.")
    parser.add_argument("--revision", default=DATA_REVISION, help="Hugging Face revision; ignored for local input.")
    parser.add_argument("--local_save_dir", "--local_dir", default="~/verl/data/deepscaler")
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face dataset cache directory.")
    parser.add_argument("--test_size", type=int, default=500, help="Number of unique problems reserved for validation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    local_path = None
    if args.local_dataset_path:
        local_path = Path(args.local_dataset_path).expanduser()
        if local_path.is_dir():
            local_path = local_path / "deepscaler.json"
        if not local_path.is_file():
            raise FileNotFoundError(f"DeepScaleR input not found: {local_path}")
        loaders = {".json": "json", ".jsonl": "json", ".parquet": "parquet"}
        if local_path.suffix.lower() not in loaders:
            raise ValueError("DeepScaleR input must be raw JSON, JSONL, or Parquet")
        dataset = datasets.load_dataset(
            loaders[local_path.suffix.lower()],
            data_files=str(local_path),
            split="train",
            cache_dir=args.cache_dir,
        )
    else:
        dataset = datasets.load_dataset(DATA_SOURCE, revision=args.revision, split="train", cache_dir=args.cache_dir)

    splits, stats = prepare_splits(dataset, test_size=args.test_size, seed=args.seed)
    destination = Path(args.local_save_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    for split, dataset in splits.items():
        dataset.to_parquet(str(destination / f"{split}.parquet"))
    manifest = {
        "data_source": DATA_SOURCE,
        "revision": None if local_path else args.revision,
        "local_dataset_path": str(local_path.resolve()) if local_path else None,
        "seed": args.seed,
        "test_size": args.test_size,
        "instruction": INSTRUCTION,
        **stats,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
