from __future__ import annotations

from src.prompt_ab import summarize_ab


def test_summarize_ab():
    results = [
        {"system_idx": 0, "prompt_idx": 0, "heuristic_score": 0.5},
        {"system_idx": 0, "prompt_idx": 0, "heuristic_score": 0.7},
        {"system_idx": 0, "prompt_idx": 1, "heuristic_score": 0.3},
    ]
    s = summarize_ab(results)
    assert s["best"] == "s0_p0"
    assert s["summary"]["s0_p0"] == 0.6
