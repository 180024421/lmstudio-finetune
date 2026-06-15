from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

MIN_CFG = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "output_dir": "output/test_run",
    "dataset": {
        "train_file": "data/examples/train.jsonl",
        "eval_file": "data/examples/eval.jsonl",
        "chat_template": "qwen",
    },
    "lora": {"name": "test-pipeline"},
}


@patch("src.pipeline.register_experiment")
def test_pipeline_skip_all_steps(mock_register, run_full_pipeline):
    result = run_full_pipeline(MIN_CFG, skip_train=True, skip_export=True, skip_eval=True)
    assert result["validate"]["ok"] is True
    assert "adapter" not in result
    assert "export" not in result
    mock_register.assert_called_once()


@patch("src.pipeline.register_experiment")
@patch("src.pipeline.export_all", return_value={"merged": Path("output/merged")})
@patch("src.pipeline.run_train", return_value=Path("output/test_run/adapter"))
def test_pipeline_with_train_and_export(mock_train, mock_export, mock_register, run_full_pipeline):
    result = run_full_pipeline(MIN_CFG, skip_train=False, skip_export=False, skip_eval=True)
    assert result["adapter"] == str(Path("output/test_run/adapter"))
    assert "export" in result
    mock_train.assert_called_once()
    mock_export.assert_called_once()


def test_pipeline_invalid_config(run_full_pipeline):
    bad_cfg = {"base_model": "", "dataset": {}}
    with pytest.raises(ValueError, match="配置错误"):
        run_full_pipeline(bad_cfg, skip_train=True, skip_export=True, skip_eval=True)


@patch("src.pipeline.register_experiment")
@patch("src.eval_runner.run_eval_judge")
def test_pipeline_with_eval_judge(mock_eval, mock_register, run_full_pipeline):
    mock_eval.return_value = MagicMock(avg_score=0.85)
    cfg = {**MIN_CFG, "pipeline": {"use_judge": True, "eval_samples": 3, "regression": False}}
    result = run_full_pipeline(cfg, skip_train=True, skip_export=True, skip_eval=False)
    assert result["eval_score"] == 0.85
    mock_eval.assert_called_once()
