#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_quality import filter_file
from src.pii_filter import redact_row
from src.data_io import load_rows, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="质量过滤 + 可选 PII 脱敏")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--rejected", default="data/rejected.jsonl")
    p.add_argument("--min-score", type=float, default=0.6)
    p.add_argument("--pii", action="store_true")
    args = p.parse_args()
    inp = Path(args.input)
    if args.pii:
        rows = [redact_row(r) for r in load_rows(inp)]
        tmp = inp.parent / "_pii_redacted.jsonl"
        write_jsonl(tmp, rows)
        inp = tmp
    stats = filter_file(inp, Path(args.output), Path(args.rejected), min_score=args.min_score)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
