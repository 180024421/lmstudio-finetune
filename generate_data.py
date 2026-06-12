#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.data_generator import generate_from_files
from src.lmstudio_client import check_lm_studio


def main() -> None:
    p = argparse.ArgumentParser(description="用 LM Studio 从文档生成训练数据")
    p.add_argument("inputs", nargs="+", help="txt/md 文件或目录")
    p.add_argument("-o", "--output", default="data/generated.jsonl")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    if not check_lm_studio(cfg):
        raise SystemExit("LM Studio 未连接")
    paths: list[Path] = []
    for inp in args.inputs:
        pth = Path(inp)
        if pth.is_dir():
            paths.extend(sorted(list(pth.glob("**/*.txt")) + list(pth.glob("**/*.md"))))
        else:
            paths.append(pth)
    generate_from_files(paths, Path(args.output), cfg)


if __name__ == "__main__":
    main()
