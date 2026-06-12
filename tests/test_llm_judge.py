from __future__ import annotations

from src.llm_judge import _parse_judge_json


def test_parse_judge():
    score, reason = _parse_judge_json('{"score": 4, "reason": "基本正确"}')
    assert score == 4
    assert "正确" in reason
