#!/usr/bin/env python3
"""SFT → DPO 串联对齐流水线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.alignment_pipeline import run_sft_dpo_pipeline
from src.config_loader import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="SFT → DPO 对齐流水线")
    p.add_argument("--config", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-dpo", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()
    cfg = load_config(Path(args.config) if args.config else None, profile=args.profile)
    result = run_sft_dpo_pipeline(
        cfg,
        skip_sft=args.skip_sft,
        skip_dpo=args.skip_dpo,
        skip_export=args.skip_export,
        skip_eval=args.skip_eval,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
