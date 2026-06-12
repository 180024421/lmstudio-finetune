from __future__ import annotations

import json
from pathlib import Path

from src.data_merge import merge_jsonl_files


def test_merge_dedup(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    out = tmp_path / "out.jsonl"
    a.write_text(json.dumps({"instruction": "q1", "output": "a1"}) + "\n", encoding="utf-8")
    b.write_text(json.dumps({"instruction": "q2", "output": "a2"}) + "\n", encoding="utf-8")
    stats = merge_jsonl_files([a, b], out, dedup=False)
    assert stats["total"] == 2
    stats2 = merge_jsonl_files([a, a, b], out, dedup=False)
    assert stats2["exact_duplicates_skipped"] == 1
