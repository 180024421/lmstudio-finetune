from __future__ import annotations

from pathlib import Path

from src.dataset import load_jsonl


def test_load_jsonl():
    ds = load_jsonl(Path(__file__).resolve().parent.parent / "data/examples/train.jsonl")
    assert len(ds) >= 2
    assert "text" in ds[0]
    assert "<|im_start|>" in ds[0]["text"]
