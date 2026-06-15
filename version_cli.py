#!/usr/bin/env python3
"""数据集版本登记与查询。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_version import register_dataset_version, versions_markdown


def main() -> None:
    p = argparse.ArgumentParser(description="数据集版本管理")
    sub = p.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("register", help="登记版本")
    reg.add_argument("file", help="JSONL 路径")
    reg.add_argument("--note", default="")
    sub.add_parser("list", help="列出版本")
    args = p.parse_args()

    if args.cmd == "register":
        entry = register_dataset_version(Path(args.file), note=args.note)
        print(json.dumps(entry, ensure_ascii=False, indent=2))
    else:
        print(versions_markdown())


if __name__ == "__main__":
    main()
