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
    parser.add_argument("--all", action="store_true", help="合并 + 多量化 GGUF + Modelfile")
    parser.add_argument("--quant", default=None, help="GGUF 量化: f16|q8_0|q4_k_m|q4_0")
    parser.add_argument("--lora", default=None, help="LoRA 注册名")
    args = parser.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    if args.lora:
        from src.lora_manager import resolve_adapter

        merge_lora(cfg, adapter_path=resolve_adapter(args.lora))
    elif args.all:
        from src.export_model import export_all

        export_all(cfg)
    else:
        merge_lora(cfg)
        if args.gguf:
            convert_to_gguf(cfg, quant=args.quant)


if __name__ == "__main__":
    main()
