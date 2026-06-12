#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.deploy_vllm import launch_vllm


def main() -> None:
    p = argparse.ArgumentParser(description="启动 vLLM OpenAI 兼容服务")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    launch_vllm(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
