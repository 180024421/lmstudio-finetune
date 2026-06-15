#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.config_loader import load_config
from src.lora_merge import merge_multiple_loras


def main() -> None:
    p = argparse.ArgumentParser(description="合并多个 LoRA 到同一基座")
    p.add_argument("names", nargs="+", help="LoRA 注册名，按顺序合并")
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()
    cfg = load_config()
    from pathlib import Path

    out = Path(args.output) if args.output else None
    path = merge_multiple_loras(args.names, cfg, output_dir=out)
    print(path)


if __name__ == "__main__":
    main()
