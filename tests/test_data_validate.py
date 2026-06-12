from __future__ import annotations

import json
from pathlib import Path

from src.data_validate import validate_jsonl


def test_validate_ok(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}) + "\n",
        encoding="utf-8",
    )
    r = validate_jsonl(p)
    assert r.ok
    assert r.total == 1


def test_validate_missing_output(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"instruction": "x", "input": "", "output": ""}) + "\n", encoding="utf-8")
    r = validate_jsonl(p)
    assert not r.ok
