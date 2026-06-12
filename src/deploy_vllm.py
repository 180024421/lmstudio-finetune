from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config

console = Console()


def build_vllm_command(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_config()
    vcfg = cfg.get("vllm") or {}
    merged = ROOT / (cfg.get("export") or {}).get("merged_dir", "output/merged")
    model = vcfg.get("model_path") or str(merged)
    port = int(vcfg.get("port", 8001))
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--dtype", vcfg.get("dtype", "auto"),
    ]
    if vcfg.get("max_model_len"):
        cmd.extend(["--max-model-len", str(int(vcfg["max_model_len"]))])
    if vcfg.get("gpu_memory_utilization"):
        cmd.extend(["--gpu-memory-utilization", str(float(vcfg["gpu_memory_utilization"]))])
    return cmd


def launch_vllm(cfg: dict[str, Any] | None = None, *, dry_run: bool = False) -> str:
    cmd = build_vllm_command(cfg)
    line = " ".join(cmd)
    console.print(f"[cyan]vLLM 命令[/cyan]\n{line}")
    if dry_run:
        return line
    console.print("[yellow]需已安装: pip install vllm[/yellow]")
    subprocess.run(cmd, check=False)
    return line
