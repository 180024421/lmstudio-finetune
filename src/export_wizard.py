"""GGUF 导出向导与校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config_loader import ROOT, load_config


def check_export_readiness(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    ecfg = cfg.get("export") or {}
    out_dir = ROOT / cfg.get("output_dir", "output/run1")
    adapter = out_dir / "adapter"
    merged = ROOT / ecfg.get("merged_dir", "output/merged")
    llama_dir = ecfg.get("llama_cpp_dir", "")
    llama_ok = False
    convert_script = None
    if llama_dir:
        for name in ("convert_hf_to_gguf.py", "convert.py"):
            p = Path(llama_dir) / name
            if p.exists():
                llama_ok = True
                convert_script = str(p)
                break

    gguf_files: list[dict[str, Any]] = []
    gguf_out = ROOT / ecfg.get("gguf_out", "output/model.gguf")
    candidates = [gguf_out] + list(ROOT.glob("output/model*.gguf"))
    seen: set[str] = set()
    for p in candidates:
        ps = str(p.resolve())
        if ps in seen or not p.exists():
            continue
        seen.add(ps)
        size_mb = p.stat().st_size / (1024 * 1024)
        gguf_files.append({"path": str(p), "size_mb": round(size_mb, 2), "valid": size_mb > 1})

    return {
        "adapter_exists": adapter.exists(),
        "adapter_path": str(adapter),
        "merged_exists": merged.exists(),
        "llama_cpp_configured": bool(llama_dir),
        "llama_cpp_ok": llama_ok,
        "convert_script": convert_script,
        "gguf_files": gguf_files,
        "ready_to_export": adapter.exists() or merged.exists(),
        "ready_for_gguf": llama_ok and (adapter.exists() or merged.exists()),
    }


def format_export_wizard_markdown(info: dict[str, Any]) -> str:
    lines = [
        "## 导出向导",
        "",
        f"- LoRA adapter: {'有' if info['adapter_exists'] else '无'} → `{info['adapter_path']}`",
        f"- 合并目录: {'有' if info['merged_exists'] else '无'}",
        f"- llama.cpp: {'已配置' if info['llama_cpp_ok'] else '未配置或脚本缺失'}",
    ]
    if info.get("convert_script"):
        lines.append(f"- 转换脚本: `{info['convert_script']}`")
    if info.get("gguf_files"):
        lines.append("", "### GGUF 文件")
        for g in info["gguf_files"]:
            mark = "OK" if g["valid"] else "过小/异常"
            lines.append(f"- {g['path']} ({g['size_mb']} MB) [{mark}]")
    else:
        lines.append("", "_尚未生成 GGUF_")
    if not info["llama_cpp_ok"]:
        lines.extend(["", "> 请在 config.yaml 设置 `export.llama_cpp_dir` 指向 llama.cpp 根目录"])
    return "\n".join(lines)
