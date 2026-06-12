#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.hf_download import download_model


def main() -> None:
    p = argparse.ArgumentParser(description="预下载 HuggingFace 基座模型（支持镜像）")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None)
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    model_id = args.model or cfg.get("base_model")
    download_model(model_id, cfg)


if __name__ == "__main__":
    main()
