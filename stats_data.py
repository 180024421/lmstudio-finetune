#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.data_stats import compute_stats


def main() -> None:
    p = argparse.ArgumentParser(description="数据集统计")
    p.add_argument("files", nargs="*", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    template = cfg.get("dataset", {}).get("chat_template", "qwen")
    files = [Path(f) for f in args.files] if args.files else [
        Path(cfg.get("dataset", {}).get("train_file", "data/examples/train.jsonl")),
    ]
    for f in files:
        print(json.dumps(compute_stats(f, template=template), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
