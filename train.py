#!/usr/bin/env python3
"""QLoRA 微调入口。"""

from __future__ import annotations

import argparse

from src.config_loader import load_config
from src.train_lora import run_train


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 QLoRA 微调")
    parser.add_argument("--config", default=None, help="config.yaml 路径")
    args = parser.parse_args()
    from pathlib import Path
    cfg = load_config(Path(args.config)) if args.config else load_config()
    run_train(cfg)


if __name__ == "__main__":
    main()
