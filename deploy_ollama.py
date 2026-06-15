#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.config_loader import load_config
from src.deploy_ollama import deploy_to_ollama


def main() -> None:
    p = argparse.ArgumentParser(description="部署到 Ollama")
    p.add_argument("--name", default="")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    from pathlib import Path

    cfg = load_config(Path(args.config)) if args.config else load_config()
    deploy_to_ollama(cfg, model_name=args.name)


if __name__ == "__main__":
    main()
