from __future__ import annotations

from src.data_dedup import deduplicate_rows


def test_dedup_removes_similar():
    rows = [
        {"instruction": "Q", "output": "你好世界测试样本A"},
        {"instruction": "Q2", "output": "你好世界测试样本A"},  # near dup
        {"instruction": "Q3", "output": "完全不同的话题内容"},
    ]
    kept, removed = deduplicate_rows(rows, threshold=0.8)
    assert removed >= 1
    assert len(kept) >= 1
