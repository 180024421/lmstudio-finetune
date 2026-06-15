from __future__ import annotations

from unittest.mock import patch

from src.doctor import format_doctor_markdown, run_doctor


@patch("src.doctor.check_lm_studio", return_value=False)
@patch("src.doctor._check_gpu", return_value={"ok": False, "error": "测试环境无 GPU"})
def test_doctor_runs(_mock_gpu, _mock_lm):
    report = run_doctor(
        cfg={
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "dataset": {"train_file": "data/examples/train.jsonl"},
        }
    )
    assert "checks" in report
    assert "python" in report
    assert "vram" in report
    md = format_doctor_markdown(report)
    assert "环境诊断" in md
    assert "显存估算" in md
