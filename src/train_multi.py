from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config

console = Console()


def build_torchrun_command(
    cfg: dict[str, Any] | None = None,
    *,
    nproc: int | None = None,
    script: str = "train.py",
    extra_args: list[str] | None = None,
) -> list[str]:
    cfg = cfg or load_config()
    mcfg = cfg.get("multi_gpu") or {}
    n = nproc or int(mcfg.get("nproc", 0)) or _detect_gpu_count()
    if n < 2:
        raise ValueError("多卡训练需要至少 2 张 GPU，或显式设置 multi_gpu.nproc >= 2")

    torchrun = shutil.which("torchrun")
    if not torchrun:
        # fallback: python -m torch.distributed.run
        py = sys.executable
        cmd = [
            py,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={n}",
            str(ROOT / script),
        ]
    else:
        cmd = [torchrun, f"--nproc_per_node={n}", str(ROOT / script)]

    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _detect_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def launch_multi_gpu_train(
    cfg: dict[str, Any] | None = None,
    *,
    nproc: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> int | list[str]:
    extra: list[str] = []
    if resume:
        extra.append("--resume")
    cmd = build_torchrun_command(cfg, nproc=nproc, extra_args=extra)
    if dry_run:
        return cmd

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    console.print(f"[cyan]启动多卡训练:[/cyan] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    return proc.returncode
