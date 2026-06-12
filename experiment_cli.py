#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from src.experiment_registry import compare_experiments, experiments_markdown, list_experiments, load_metrics_tail


def main() -> None:
    p = argparse.ArgumentParser(description="实验记录与对比")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出实验")
    sub.add_parser("compare", help="对比实验")
    sub.add_parser("markdown", help="Markdown 表格")
    tail = sub.add_parser("metrics", help="查看 metrics 尾部")
    tail.add_argument("path")
    args = p.parse_args()

    if args.cmd == "list":
        print(json.dumps(list_experiments(), ensure_ascii=False, indent=2))
    elif args.cmd == "compare":
        print(json.dumps(compare_experiments(), ensure_ascii=False, indent=2))
    elif args.cmd == "markdown":
        print(experiments_markdown())
    elif args.cmd == "metrics":
        print(json.dumps(load_metrics_tail(args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
