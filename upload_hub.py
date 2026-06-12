#!/usr/bin/env python3
"""上传模型到 HuggingFace Hub。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.hub_upload import upload_to_hub


def main() -> None:
    p = argparse.ArgumentParser(description="上传到 HuggingFace Hub")
    p.add_argument("--config", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--repo-id", default="")
    p.add_argument("--folder", default=None, help="本地目录，默认 merged 或 adapter")
    p.add_argument("--public", action="store_true", help="公开仓库")
    p.add_argument("-m", "--message", default="Upload from lmstudio-finetune")
    args = p.parse_args()
    cfg = load_config(Path(args.config) if args.config else None, profile=args.profile)
    folder = Path(args.folder) if args.folder else None
    url = upload_to_hub(
        cfg,
        repo_id=args.repo_id,
        folder=folder,
        private=not args.public,
        commit_message=args.message,
    )
    print(url)


if __name__ == "__main__":
    main()
