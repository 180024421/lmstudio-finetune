#!/usr/bin/env python3
"""多卡训练启动（torchrun）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config_loader import load_config
from src.train_multi import build_torchrun_command, launch_multi_gpu_train


def main() -> None:
    p = argparse.ArgumentParser(description="多卡 QLoRA 训练 (torchrun)")
    p.add_argument("--config", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("-n", "--nproc", type=int, default=0, help="GPU 数量，0=自动检测")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = p.parse_args()
    cfg = load_config(Path(args.config) if args.config else None, profile=args.profile)
    nproc = args.nproc or None
    if args.dry_run:
        cmd = build_torchrun_command(cfg, nproc=nproc, extra_args=(["--resume"] if args.resume else []))
        print(" ".join(cmd))
        return
    code = launch_multi_gpu_train(cfg, nproc=nproc, resume=args.resume)
    sys.exit(code)


if __name__ == "__main__":
    main()
