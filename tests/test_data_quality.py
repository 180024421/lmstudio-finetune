from __future__ import annotations

from src.data_quality import score_row, filter_rows


def test_good_row():
    s, issues = score_row({"instruction": "Q", "output": "这是一个足够长的有效回答。"})
    assert s >= 0.6
    assert not issues or "过短" not in str(issues)


def test_bad_row_filtered():
    kept, rej = filter_rows([{"instruction": "Q", "output": ""}], min_score=0.6)
    assert len(kept) == 0
    assert len(rej) == 1
