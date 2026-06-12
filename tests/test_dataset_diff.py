from __future__ import annotations

import json
from pathlib import Path

from src.dataset_diff import diff_datasets


def test_diff(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(
        json.dumps({"instruction": "q1", "output": "a1"}) + "\n"
        + json.dumps({"instruction": "q2", "output": "a2"}) + "\n",
        encoding="utf-8",
    )
    b.write_text(json.dumps({"instruction": "q1", "output": "a1"}) + "\n", encoding="utf-8")
    r = diff_datasets(a, b)
    assert r["common"] == 1
    assert r["only_in_a"] == 1
    assert r["only_in_b"] == 0
