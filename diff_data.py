#!/usr/bin/env python3
"""对比两个 JSONL 数据集差异。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset_diff import diff_datasets


def main() -> None:
    p = argparse.ArgumentParser(description="对比两个 JSONL 数据集")
    p.add_argument("a", help="数据集 A")
    p.add_argument("b", help="数据集 B")
    p.add_argument("--limit", type=int, default=20, help="差异样本预览条数")
    args = p.parse_args()
    result = diff_datasets(Path(args.a), Path(args.b), sample_limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
