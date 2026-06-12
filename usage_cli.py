#!/usr/bin/env python3
"""API 调用成本统计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.api_usage import summarize_usage, usage_markdown
from src.config_loader import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="LM Studio API 用量统计")
    p.add_argument("--config", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()
    cfg = load_config(Path(args.config) if args.config else None, profile=args.profile)
    if args.markdown:
        print(usage_markdown(cfg))
    else:
        print(json.dumps(summarize_usage(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
