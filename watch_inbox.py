#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from src.config_loader import load_config
from src.inbox_watcher import process_inbox, watch_loop


def main() -> None:
    p = argparse.ArgumentParser(description="监视 data/inbox 自动合并训练集")
    p.add_argument("--once", action="store_true", help="只处理一次")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    from pathlib import Path

    cfg = load_config(Path(args.config)) if args.config else load_config()
    if args.once:
        print(json.dumps(process_inbox(cfg), ensure_ascii=False, indent=2))
    else:
        watch_loop(cfg)


if __name__ == "__main__":
    main()
