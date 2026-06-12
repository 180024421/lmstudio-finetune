#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from src.config_loader import load_config
from src.pipeline import run_full_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="一键流水线：校验→训练→导出→评测")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    from pathlib import Path
    cfg = load_config(Path(args.config)) if args.config else load_config()
    result = run_full_pipeline(
        cfg,
        skip_train=args.skip_train,
        skip_export=args.skip_export,
        skip_eval=args.skip_eval,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
