#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.prompt_ab import print_ab_summary, run_prompt_ab, save_ab_report, summarize_ab


def main() -> None:
    p = argparse.ArgumentParser(description="System Prompt / 前缀 A/B 测试")
    p.add_argument("--questions", nargs="+", required=True)
    p.add_argument("--prefixes", nargs="*", default=["", "请简洁回答："])
    p.add_argument("--systems", nargs="*", default=[""])
    p.add_argument("-o", "--output", default="output/prompt_ab.json")
    args = p.parse_args()
    cfg = load_config()
    results = run_prompt_ab(args.prefixes, args.questions, system_prompts=args.systems, cfg=cfg)
    save_ab_report(results, Path(args.output))
    print_ab_summary(results)
    print(json.dumps(summarize_ab(results), ensure_ascii=False))


if __name__ == "__main__":
    main()
