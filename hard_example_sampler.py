#!/usr/bin/env python3
"""难例采样：从评测报告回流训练集。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.hard_example_sampler import sample_hard_examples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_report", type=Path, default=Path("output/eval_report.json"), nargs="?")
    p.add_argument("-o", "--output", type=Path, default=Path("data/hard_examples.jsonl"))
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--max", type=int, default=50, dest="max_n")
    args = p.parse_args()
    r = sample_hard_examples(args.eval_report, threshold=args.threshold, out_path=args.output, max_n=args.max_n)
    print(r)


if __name__ == "__main__":
    main()
