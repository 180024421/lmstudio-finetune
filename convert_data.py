#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_convert import convert_file


def main() -> None:
    p = argparse.ArgumentParser(description="转换 ShareGPT / CSV / FAQ Markdown → JSONL")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-f", "--format", required=True, choices=["sharegpt", "csv", "faq_md", "rag_csv", "rag_md"])
    p.add_argument("--instruction-col", default="instruction")
    p.add_argument("--input-col", default="input")
    p.add_argument("--output-col", default="output")
    args = p.parse_args()
    n = convert_file(
        Path(args.input),
        Path(args.output),
        args.format,
        instruction_col=args.instruction_col,
        input_col=args.input_col,
        output_col=args.output_col,
    )
    print(f"已转换 {n} 条 → {args.output}")


if __name__ == "__main__":
    main()
