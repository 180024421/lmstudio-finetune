from __future__ import annotations

import json
from pathlib import Path

from src.train_report import build_train_report, report_to_markdown


def test_train_report(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 2.0}) + "\n" + json.dumps({"step": 2, "loss": 1.0}) + "\n",
        encoding="utf-8",
    )
    report = build_train_report({"base_model": "m", "output_dir": str(run_dir)}, run_dir=run_dir)
    assert report["final_loss"] == 1.0
    md = report_to_markdown(report)
    assert "训练报告" in md
