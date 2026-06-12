#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.data_dedup import deduplicate_file


def main() -> None:
    p = argparse.ArgumentParser(description="语义近似去重 JSONL")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--threshold", type=float, default=0.85)
    args = p.parse_args()
    cfg = load_config()
    template = cfg.get("dataset", {}).get("chat_template", "qwen")
    stats = deduplicate_file(Path(args.input), Path(args.output), template=template, threshold=args.threshold)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
