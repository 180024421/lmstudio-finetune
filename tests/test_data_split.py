from __future__ import annotations

import json
from pathlib import Path

from src.data_split import split_dataset, split_summary


def test_split(tmp_path: Path):
    src = tmp_path / "all.jsonl"
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(10)]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    paths = split_dataset(src, tmp_path / "out", train_ratio=0.8, eval_ratio=0.1, test_ratio=0.1, seed=1)
    counts = split_summary(paths)
    assert counts["train"] == 8
    assert counts["eval"] == 1
    assert counts["test"] == 1
