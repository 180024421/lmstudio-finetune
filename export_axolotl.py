#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.export_axolotl import to_axolotl_yaml


def main() -> None:
    p = argparse.ArgumentParser(description="导出 Axolotl 训练配置")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    out = to_axolotl_yaml(cfg, Path(args.output) if args.output else None)
    print(out)


if __name__ == "__main__":
    main()
