#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config_loader import load_config
from src.data_validate import validate_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="校验 JSONL 训练数据")
    p.add_argument("file", nargs="?", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--register-version", action="store_true", help="校验通过后登记数据集版本")
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    dcfg = cfg.get("dataset") or {}
    path = Path(args.file) if args.file else Path(dcfg.get("train_file", "data/examples/train.jsonl"))
    report = validate_jsonl(path, template=dcfg.get("chat_template", "qwen"))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    for issue in report.errors[:20]:
        print(f"ERROR L{issue.line}: {issue.message}", file=sys.stderr)
    for issue in report.warnings[:20]:
        print(f"WARN  L{issue.line}: {issue.message}", file=sys.stderr)
    if report.ok and args.register_version:
        from src.data_version import register_dataset_version
        ver = register_dataset_version(path)
        print(f"已登记版本 fingerprint={ver['fingerprint']} count={ver['count']}")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
