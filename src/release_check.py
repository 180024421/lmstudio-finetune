"""发布检查清单：训练 → 导出 → GGUF → LM Studio → Benchmark。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_loader import ROOT, load_config
from .lmstudio_health import run_lmstudio_health


def _step(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def run_release_check(cfg: dict[str, Any] | None = None, *, skip_lmstudio: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    steps: list[dict[str, Any]] = []

    train_file = ROOT / (cfg.get("dataset") or {}).get("train_file", "data/examples/train.jsonl")
    steps.append(_step("训练数据", train_file.exists(), str(train_file)))

    out_dir = ROOT / cfg.get("output_dir", "output/run1")
    adapter = out_dir / "adapter"
    steps.append(_step("LoRA adapter", adapter.exists(), str(adapter)))

    ecfg = cfg.get("export") or {}
    merged = ROOT / ecfg.get("merged_dir", "output/merged")
    steps.append(_step("合并模型", merged.exists(), str(merged)))

    gguf_out = ROOT / ecfg.get("gguf_out", "output/model.gguf")
    gguf_ok = False
    gguf_detail = str(gguf_out)
    if gguf_out.exists():
        size_mb = gguf_out.stat().st_size // 1024 // 1024
        gguf_ok = gguf_out.stat().st_size > 1024 * 1024
        gguf_detail = f"{gguf_out} ({size_mb}MB)"
    steps.append(_step("GGUF 文件", gguf_ok, gguf_detail))

    card = merged / "MODEL_CARD.md"
    steps.append(_step("Model Card", card.exists(), str(card)))

    if not skip_lmstudio:
        health = run_lmstudio_health(cfg)
        steps.append(_step("LM Studio API", health["connected"], ", ".join(health["models"]) or "未加载模型"))

    eval_report = ROOT / "output" / "eval_report.json"
    steps.append(_step("评测报告", eval_report.exists(), str(eval_report)))

    ok = all(s["ok"] for s in steps)
    return {"ok": ok, "steps": steps}


def format_release_markdown(report: dict[str, Any]) -> str:
    lines = ["# 发布检查清单", "", f"总体: {'通过' if report['ok'] else '未通过'}", ""]
    for s in report["steps"]:
        mark = "x" if s["ok"] else " "
        lines.append(f"- [{mark}] **{s['name']}**: {s.get('detail', '')}")
    return "\n".join(lines)
