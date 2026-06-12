#!/usr/bin/env python3
"""合并 LoRA 并可选转 GGUF。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.export_model import convert_to_gguf, merge_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--gguf", action="store_true", help="尝试转 GGUF")
    args = parser.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    merge_lora(cfg)
    if args.gguf:
        convert_to_gguf(cfg)


if __name__ == "__main__":
    main()
