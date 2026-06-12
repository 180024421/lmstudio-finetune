#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.semantic_dedup import semantic_deduplicate_file


def main() -> None:
    p = argparse.ArgumentParser(description="语义 embedding 去重（无依赖时回退字符去重）")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--threshold", type=float, default=0.92)
    args = p.parse_args()
    cfg = load_config()
    template = cfg.get("dataset", {}).get("chat_template", "qwen")
    stats = semantic_deduplicate_file(Path(args.input), Path(args.output), template=template, threshold=args.threshold)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
