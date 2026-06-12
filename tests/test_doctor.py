from __future__ import annotations

from src.doctor import format_doctor_markdown, run_doctor


def test_doctor_runs():
    report = run_doctor(cfg={"base_model": "test/model", "dataset": {"train_file": "data/examples/train.jsonl"}})
    assert "checks" in report
    assert "python" in report
    md = format_doctor_markdown(report)
    assert "环境诊断" in md
