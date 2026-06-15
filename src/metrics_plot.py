from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_loader import ROOT


def load_metrics_series(metrics_path: Path) -> dict[str, list[tuple[int, float]]]:
    p = metrics_path if metrics_path.is_absolute() else ROOT / metrics_path
    series: dict[str, list[tuple[int, float]]] = {}
    if not p.exists():
        return series
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = int(row.get("step", 0))
        for k, v in row.items():
            if k == "step" or not isinstance(v, (int, float)):
                continue
            series.setdefault(k, []).append((step, float(v)))
    return series


def to_plot_dataframe(metrics_path: Path):
    """返回 Gradio LinePlot 可用的 DataFrame。"""
    try:
        import pandas as pd
    except ImportError:
        return None
    series = load_metrics_series(metrics_path)
    rows: list[dict[str, Any]] = []
    for name, points in series.items():
        for step, val in points:
            rows.append({"step": step, "metric": name, "value": val})
    if not rows:
        return None
    return pd.DataFrame(rows)


def to_gradio_plots(metrics_path: Path) -> list[dict[str, Any]]:
    series = load_metrics_series(metrics_path)
    plots = []
    for name, points in series.items():
        if not points:
            continue
        plots.append(
            {
                "title": name,
                "data": [[p[0], p[1]] for p in points],
                "headers": ["step", name],
            }
        )
    return plots
