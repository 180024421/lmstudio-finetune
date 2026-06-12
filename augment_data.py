#!/usr/bin/env python3
"""通过 LM Studio 对训练数据进行释义增强。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.data_augment import augment_file


def main() -> None:
    p = argparse.ArgumentParser(description="LM Studio 数据增强（释义变体）")
    p.add_argument("input", help="输入 JSONL")
    p.add_argument("-o", "--output", required=True, help="输出 JSONL")
    p.add_argument("-n", "--variants", type=int, default=2, help="每条样本生成变体数")
    p.add_argument("--max-rows", type=int, default=0, help="只处理前 N 条源样本")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    stats = augment_file(
        Path(args.input),
        Path(args.output),
        cfg=cfg,
        variants_per_row=args.variants,
        max_rows=args.max_rows,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
