#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.distill_generator import distill_from_file, distill_from_topics


def main() -> None:
    p = argparse.ArgumentParser(description="教师模型蒸馏造数")
    p.add_argument("input", nargs="?", help="文档 md/txt")
    p.add_argument("-o", "--output", default="data/distilled.jsonl")
    p.add_argument("--topic", action="append", dest="topics", help="直接指定知识点")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    if args.topics:
        from src.data_io import write_jsonl

        rows = distill_from_topics(args.topics, cfg)
        write_jsonl(Path(args.output), rows)
        print(len(rows))
    elif args.input:
        distill_from_file(Path(args.input), Path(args.output), cfg)
    else:
        raise SystemExit("需要 input 文件或 --topic")


if __name__ == "__main__":
    main()
