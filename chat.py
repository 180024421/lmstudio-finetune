#!/usr/bin/env python3
"""测试 LM Studio 本地 API。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config_loader import load_config
from src.lmstudio_client import chat, check_lm_studio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    if not check_lm_studio(cfg):
        print("LM Studio 未连接。请启动 LM Studio → 加载模型 → 开启 Local Server (1234)", file=sys.stderr)
        sys.exit(1)
    print(chat(args.prompt, cfg))


if __name__ == "__main__":
    main()
