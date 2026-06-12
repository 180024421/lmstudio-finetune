#!/usr/bin/env python3
"""抓取网页文档并转为 Markdown，可接 generate_data 造数。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.data_generator import generate_from_files
from src.doc_crawler import crawl_to_markdown_files


def main() -> None:
    p = argparse.ArgumentParser(description="抓取网页 → Markdown → 可选造数")
    p.add_argument("urls", nargs="+", help="起始 URL")
    p.add_argument("-o", "--output-dir", default="data/crawled", help="Markdown 输出目录")
    p.add_argument("--max-pages", type=int, default=0)
    p.add_argument("--generate", action="store_true", help="抓取后调用 LM Studio 造数")
    p.add_argument("--jsonl-out", default="data/crawled_train.jsonl")
    p.add_argument("--config", default=None)
    p.add_argument("--profile", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config) if args.config else None, profile=args.profile)
    ccfg = cfg.get("crawler") or {}
    max_pages = args.max_pages or int(ccfg.get("max_pages", 10))
    stats = crawl_to_markdown_files(
        args.urls,
        Path(args.output_dir),
        max_pages=max_pages,
        same_domain_only=bool(ccfg.get("same_domain_only", True)),
        delay_seconds=float(ccfg.get("delay_seconds", 1.0)),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.generate and stats.get("files"):
        n = generate_from_files([Path(f) for f in stats["files"]], Path(args.jsonl_out), cfg)
        print(f"已造数 {n} 条 → {args.jsonl_out}")


if __name__ == "__main__":
    main()
