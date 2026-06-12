#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.data_quality_judge import filter_file_with_judge


def main() -> None:
    p = argparse.ArgumentParser(description="规则 + LLM 评审双重质检")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--rejected", default="data/judge_rejected.jsonl")
    args = p.parse_args()
    stats = filter_file_with_judge(
        Path(args.input), Path(args.output), Path(args.rejected), load_config()
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
