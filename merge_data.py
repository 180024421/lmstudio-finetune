#!/usr/bin/env python3
"""合并多个 JSONL 训练文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.data_merge import merge_jsonl_files


def main() -> None:
    p = argparse.ArgumentParser(description="合并 JSONL 数据集")
    p.add_argument("inputs", nargs="+", help="输入 JSONL 文件")
    p.add_argument("-o", "--output", required=True, help="输出路径")
    p.add_argument("--no-dedup", action="store_true", help="不做模糊去重")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    dcfg = cfg.get("dataset") or {}
    stats = merge_jsonl_files(
        [Path(x) for x in args.inputs],
        Path(args.output),
        dedup=not args.no_dedup,
        template=dcfg.get("chat_template", "qwen"),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
