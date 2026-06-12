from __future__ import annotations

from src.api_usage import record_usage, summarize_usage, usage_log_path


def test_api_usage(tmp_path, monkeypatch):
    log = tmp_path / "usage.json"
    monkeypatch.setattr("src.api_usage.usage_log_path", lambda cfg=None: log)
    record_usage(operation="test", model="m", prompt_tokens=100, completion_tokens=50)
    record_usage(operation="test", model="m", prompt_tokens=200, completion_tokens=100)
    s = summarize_usage()
    assert s["total_calls"] == 2
    assert s["total_tokens"] == 450
