from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .data_io import load_rows, write_jsonl


def split_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> dict[str, Path]:
    total = train_ratio + eval_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为 1，当前为 {total}")

    rows = load_rows(input_path)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * train_ratio)
    n_eval = int(n * eval_ratio)
    train_rows = rows[:n_train]
    eval_rows = rows[n_train : n_train + n_eval]
    test_rows = rows[n_train + n_eval :]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.jsonl",
        "eval": output_dir / "eval.jsonl",
        "test": output_dir / "test.jsonl",
    }
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["eval"], eval_rows)
    write_jsonl(paths["test"], test_rows)
    return paths


def split_summary(paths: dict[str, Path]) -> dict[str, Any]:
    from .data_io import load_rows

    return {k: len(load_rows(p)) for k, p in paths.items()}
