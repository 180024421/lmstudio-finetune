#!/usr/bin/env python3
"""LM Studio 健康检查。"""

from __future__ import annotations

import argparse

from src.lmstudio_health import format_health_markdown, run_lmstudio_health


def main() -> None:
    p = argparse.ArgumentParser(description="LM Studio 连接与加载建议")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()
    report = run_lmstudio_health()
    print(format_health_markdown(report) if args.markdown else report)
    raise SystemExit(0 if report["connected"] else 1)


if __name__ == "__main__":
    main()
