from __future__ import annotations

import json
from pathlib import Path

from src.data_sample import shuffle_and_sample


def test_shuffle_sample(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(10)]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    s = shuffle_and_sample(inp, out, seed=1, max_samples=3)
    assert s["kept"] == 3
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 3
