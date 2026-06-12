#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.train_orpo import run_orpo


def main() -> None:
    p = argparse.ArgumentParser(description="ORPO 偏好训练")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    run_orpo(load_config(Path(args.config)) if args.config else load_config())


if __name__ == "__main__":
    main()
