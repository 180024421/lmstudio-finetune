from __future__ import annotations

import json
from pathlib import Path

from src.metrics_plot import load_metrics_series, to_gradio_plots


def test_metrics_plot(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps({"step": 1, "loss": 2.0}) + "\n" + json.dumps({"step": 2, "loss": 1.0}) + "\n",
        encoding="utf-8",
    )
    s = load_metrics_series(p)
    assert "loss" in s
    plots = to_gradio_plots(p)
    assert plots[0]["title"] == "loss"
