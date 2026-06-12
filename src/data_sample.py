from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .data_io import load_rows, write_jsonl


def shuffle_and_sample(
    path: Path,
    output: Path,
    *,
    seed: int = 42,
    max_samples: int = 0,
    fraction: float = 0.0,
) -> dict[str, Any]:
    rows = load_rows(path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    original = len(rows)

    if fraction > 0:
        n = max(1, int(len(rows) * fraction))
        rows = rows[:n]
    elif max_samples > 0:
        rows = rows[:max_samples]

    write_jsonl(output, rows)
    return {
        "input": str(path),
        "output": str(output),
        "original": original,
        "kept": len(rows),
        "seed": seed,
    }
