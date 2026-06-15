from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config_loader import ROOT, load_config
from .config_schema import validate_config
from .data_validate import validate_jsonl
from .lmstudio_client import check_lm_studio
from .vram_estimate import estimate_vram_gb, format_vram_markdown


def _check_import(name: str) -> dict[str, Any]:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "ok")
        return {"ok": True, "version": str(ver)}
    except ImportError as e:
        return {"ok": False, "error": str(e)}


def _check_gpu() -> dict[str, Any]:
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0:
                gpus = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
                return {"ok": True, "gpus": gpus}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "ok": True,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        return {"ok": False, "error": "CUDA 不可用"}
    except ImportError:
        return {"ok": False, "error": "未安装 torch，无法检测 GPU"}


def run_doctor(*, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    report: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "checks": [],
        "ok": True,
    }

    def add(name: str, ok: bool, detail: str = "", extra: dict[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {"name": name, "ok": ok, "detail": detail}
        if extra:
            entry.update(extra)
        report["checks"].append(entry)
        if not ok:
            report["ok"] = False

    venv_ok = (ROOT / ".venv").exists()
    add("虚拟环境", venv_ok, ".venv 存在" if venv_ok else "请运行 .\\run.ps1 -Setup")

    cfg_errs = validate_config(cfg)
    add("配置校验", not cfg_errs, "; ".join(cfg_errs) if cfg_errs else "通过")

    dcfg = cfg.get("dataset") or {}
    train_file = ROOT / dcfg.get("train_file", "data/examples/train.jsonl")
    if train_file.exists():
        vr = validate_jsonl(train_file, template=dcfg.get("chat_template", "qwen"))
        add("训练数据", vr.ok, f"{train_file.name} {vr.total} 条", {"errors": len(vr.errors)})
    else:
        add("训练数据", False, f"不存在: {train_file}")

    lm_ok = check_lm_studio()
    add("LM Studio", lm_ok, "已连接" if lm_ok else "未连接 http://127.0.0.1:1234")

    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "gradio"):
        info = _check_import(pkg)
        required = pkg in ("transformers", "datasets")
        add(f"包 {pkg}", info["ok"] or not required, info.get("version", info.get("error", "")))

    gpu = _check_gpu()
    add("GPU", gpu["ok"], ", ".join(gpu.get("gpus", [])) or gpu.get("error", ""), gpu)

    out_dir = ROOT / cfg.get("output_dir", "output/run1")
    writable = False
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test = out_dir / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
        writable = True
    except OSError as e:
        add("输出目录", False, str(e))
    if writable:
        add("输出目录", True, str(out_dir))

    llama = (cfg.get("export") or {}).get("llama_cpp_dir") or ""
    if llama:
        lp = Path(llama)
        add("llama.cpp", lp.exists(), str(lp))
    else:
        add("llama.cpp", True, "未配置（导出 GGUF 时需设置 export.llama_cpp_dir）")

    vram = estimate_vram_gb(cfg)
    report["vram"] = vram
    add(
        "显存估算",
        True,
        f"预估 {vram['estimated_gb']} GB，建议 ≥ {vram['recommended_gb']} GB",
        {"breakdown": vram["breakdown"]},
    )

    return report


def format_doctor_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 环境诊断",
        "",
        f"- Python: {report['python']}",
        f"- 平台: {report['platform']}",
        f"- 总体: {'[OK]' if report['ok'] else '[FAIL]'}",
        "",
        "| 检查项 | 状态 | 详情 |",
        "|--------|------|------|",
    ]
    for c in report["checks"]:
        status = "OK" if c["ok"] else "FAIL"
        lines.append(f"| {c['name']} | {status} | {c.get('detail', '')} |")
    if report.get("vram"):
        lines.extend(["", format_vram_markdown(report["vram"])])
    return "\n".join(lines)
