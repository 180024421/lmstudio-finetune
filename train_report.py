#!/usr/bin/env python3
"""生成训练报告（Markdown / HTML）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.train_report import save_train_report


def main() -> None:
    p = argparse.ArgumentParser(description="生成训练报告")
    p.add_argument("-o", "--output", default="output/train_report.md")
    p.add_argument("--run-dir", default=None, help="训练输出目录")
    p.add_argument("--html", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    run_dir = Path(args.run_dir) if args.run_dir else None
    path = save_train_report(Path(args.output), cfg, run_dir=run_dir, html=args.html)
    print(f"报告已保存: {path}")


if __name__ == "__main__":
    main()
