#!/usr/bin/env python3
"""环境与健康检查。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config_loader import load_config
from src.doctor import format_doctor_markdown, run_doctor


def main() -> None:
    p = argparse.ArgumentParser(description="环境诊断")
    p.add_argument("--config", default=None)
    p.add_argument("--markdown", action="store_true", help="输出 Markdown")
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    report = run_doctor(cfg=cfg)
    if args.markdown:
        sys.stdout.buffer.write(format_doctor_markdown(report).encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
