#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.data_split import split_dataset, split_summary


def main() -> None:
    p = argparse.ArgumentParser(description="切分 train/eval/test")
    p.add_argument("input", nargs="?", default=None)
    p.add_argument("-o", "--output", default="data")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--eval", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = load_config()
    inp = Path(args.input or cfg.get("dataset", {}).get("train_file", "data/all.jsonl"))
    paths = split_dataset(inp, Path(args.output), train_ratio=args.train, eval_ratio=args.eval, test_ratio=args.test, seed=args.seed)
    print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))
    print(json.dumps(split_summary(paths), ensure_ascii=False))


if __name__ == "__main__":
    main()
