from __future__ import annotations

from src.eval_runner import _score_response


def test_exact_match():
    s, _ = _score_response("hello", "hello")
    assert s == 1.0


def test_keywords():
    s, note = _score_response("ignored", "has foo and bar", keywords=["foo", "bar"])
    assert s > 0.5
