#!/usr/bin/env python3
"""多轮对话数据分析。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dialogue_analytics import analyze_dialogue_dataset, format_analytics_markdown


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("file", type=Path)
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()
    stats = analyze_dialogue_dataset(args.file)
    if args.markdown:
        print(format_analytics_markdown(stats))
    else:
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
