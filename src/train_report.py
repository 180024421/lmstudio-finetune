from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import ROOT, load_config
from .metrics_plot import load_metrics_series


def _read_trainer_state(run_dir: Path) -> dict[str, Any]:
    for name in ("trainer_state.json", "checkpoint-*/trainer_state.json"):
        if "*" in name:
            cps = sorted(run_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for cp in cps:
                p = cp / "trainer_state.json"
                if p.exists():
                    return json.loads(p.read_text(encoding="utf-8"))
            return {}
        p = run_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def build_train_report(
    cfg: dict[str, Any] | None = None,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    out_dir = run_dir or ROOT / cfg.get("output_dir", "output/run1")
    metrics_path = out_dir / "metrics.jsonl"
    series = load_metrics_series(metrics_path)
    state = _read_trainer_state(out_dir)

    def _last(key: str) -> float | None:
        pts = series.get(key, [])
        return pts[-1][1] if pts else None

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_model": cfg.get("base_model"),
        "output_dir": str(out_dir),
        "lora": cfg.get("lora"),
        "train_config": cfg.get("train"),
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "final_loss": _last("loss"),
        "final_eval_loss": _last("eval_loss"),
        "metrics_summary": {k: {"first": v[0][1], "last": v[-1][1], "points": len(v)} for k, v in series.items()},
    }
    return report


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 训练报告",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 基座模型: {report.get('base_model')}",
        f"- 输出目录: {report.get('output_dir')}",
        f"- 全局步数: {report.get('global_step')}",
        f"- 最终 loss: {report.get('final_loss')}",
        f"- 最终 eval_loss: {report.get('final_eval_loss')}",
        f"- 最佳 checkpoint: {report.get('best_model_checkpoint')}",
        "",
        "## 指标摘要",
        "",
    ]
    for name, info in (report.get("metrics_summary") or {}).items():
        lines.append(f"- **{name}**: {info['first']:.4f} → {info['last']:.4f} ({info['points']} 点)")
    lora = report.get("lora") or {}
    if lora:
        lines.extend(["", "## LoRA", "", f"- r={lora.get('r')} alpha={lora.get('lora_alpha')}"])
    return "\n".join(lines)


def report_to_html(report: dict[str, Any]) -> str:
    md = report_to_markdown(report)
    body = md.replace("\n", "<br>\n").replace("## ", "<h2>").replace("# ", "<h1>")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>训练报告</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:2em auto;line-height:1.6}}</style>
</head><body>{body}</body></html>"""


def save_train_report(
    output: Path,
    cfg: dict[str, Any] | None = None,
    *,
    run_dir: Path | None = None,
    html: bool = False,
) -> Path:
    report = build_train_report(cfg, run_dir=run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    if html or output.suffix.lower() == ".html":
        output.write_text(report_to_html(report), encoding="utf-8")
    else:
        output.write_text(report_to_markdown(report), encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
