#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_io import load_rows
from src.review_store import approve_index, list_pending, merge_approved_to, queue_for_review, reject_index


def main() -> None:
    p = argparse.ArgumentParser(description="人工校对队列")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="待校对列表")
    imp = sub.add_parser("import", help="导入 JSONL 到待校对")
    imp.add_argument("file")

    app = sub.add_parser("approve", help="通过一条")
    app.add_argument("index", type=int)
    app.add_argument("--edit", default=None, help="修改后的 assistant 内容")

    rej = sub.add_parser("reject", help="拒绝一条")
    rej.add_argument("index", type=int)

    mer = sub.add_parser("merge", help="合并已通过到训练集")
    mer.add_argument("-o", "--output", default="data/train.jsonl")

    args = p.parse_args()
    if args.cmd == "list":
        rows = list_pending()
        for i, r in enumerate(rows):
            q = r.get("instruction") or (r.get("messages") or [{}])[0].get("content", "")
            print(f"[{i}] {str(q)[:60]}")
        print(f"共 {len(rows)} 条待校对")
    elif args.cmd == "import":
        n = queue_for_review(load_rows(Path(args.file)))
        print(f"已加入 {n} 条")
    elif args.cmd == "approve":
        row = approve_index(args.index, edited_output=args.edit)
        print(json.dumps(row, ensure_ascii=False) if row else "无效索引")
    elif args.cmd == "reject":
        print("OK" if reject_index(args.index) else "无效索引")
    elif args.cmd == "merge":
        n = merge_approved_to(Path(args.output))
        print(f"已合并 {n} 条 → {args.output}")


if __name__ == "__main__":
    main()
