#!/usr/bin/env python3
"""打乱并可选抽样 JSONL 数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_sample import shuffle_and_sample


def main() -> None:
    p = argparse.ArgumentParser(description="打乱 / 抽样数据集")
    p.add_argument("input", help="输入 JSONL")
    p.add_argument("-o", "--output", required=True, help="输出 JSONL")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-n", "--max-samples", type=int, default=0, help="最多保留 N 条，0=全部")
    p.add_argument("-f", "--fraction", type=float, default=0.0, help="保留比例 0~1")
    args = p.parse_args()
    stats = shuffle_and_sample(
        Path(args.input),
        Path(args.output),
        seed=args.seed,
        max_samples=args.max_samples,
        fraction=args.fraction,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
