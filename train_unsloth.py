#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.train_unsloth import run_unsloth


def main() -> None:
    p = argparse.ArgumentParser(description="Unsloth 加速 QLoRA 训练")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    run_unsloth(cfg)


if __name__ == "__main__":
    main()
