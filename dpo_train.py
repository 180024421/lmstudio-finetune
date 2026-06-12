#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.train_dpo import run_dpo


def main() -> None:
    p = argparse.ArgumentParser(description="DPO 偏好对齐训练")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    run_dpo(cfg)


if __name__ == "__main__":
    main()
