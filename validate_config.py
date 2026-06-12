#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config_loader import load_config
from src.config_schema import validate_config


def main() -> None:
    p = argparse.ArgumentParser(description="校验 config.yaml")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    errs = validate_config(cfg)
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        raise SystemExit(1)
    print("配置校验通过")


if __name__ == "__main__":
    main()
