#!/usr/bin/env python3
"""QLoRA 微调入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config_loader import load_config
from src.config_schema import validate_config
from src.data_validate import validate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 QLoRA 微调")
    parser.add_argument("--config", default=None, help="config.yaml 路径")
    parser.add_argument("--profile", default=None, help="合并 config.{profile}.yaml")
    parser.add_argument("--resume", action="store_true", help="从最新 checkpoint 续训")
    parser.add_argument("--dry-run", action="store_true", help="仅校验配置与数据")
    args = parser.parse_args()
    cfg = load_config(Path(args.config) if args.config else None, profile=args.profile)

    errs = validate_config(cfg)
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        raise SystemExit(1)

    dcfg = cfg.get("dataset") or {}
    train_file = Path(dcfg.get("train_file", "data/examples/train.jsonl"))
    if not train_file.is_absolute():
        from src.config_loader import ROOT

        train_file = ROOT / train_file
    report = validate_jsonl(train_file, template=dcfg.get("chat_template", "qwen"))
    if not report.ok:
        for e in report.errors[:10]:
            print(f"数据错误 L{e.line}: {e.message}", file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run:
        print(f"校验通过：{train_file} ({report.total} 条)")
        return

    from src.train_lora import run_train

    run_train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
