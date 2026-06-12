from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config

console = Console()


def upload_to_hub(
    cfg: dict[str, Any] | None = None,
    *,
    repo_id: str = "",
    folder: Path | None = None,
    private: bool | None = None,
    commit_message: str = "Upload from lmstudio-finetune",
) -> str:
    cfg = cfg or load_config()
    hcfg = cfg.get("hub") or {}
    repo = repo_id or hcfg.get("repo_id") or ""
    if not repo:
        raise ValueError("请设置 hub.repo_id 或传入 --repo-id")

    token = hcfg.get("token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("需要 HF_TOKEN 环境变量或 hub.token")

    if folder is None:
        if hcfg.get("upload_merged", True):
            folder = ROOT / (cfg.get("export") or {}).get("merged_dir", "output/merged")
        else:
            folder = ROOT / cfg.get("output_dir", "output/run1") / "adapter"

    folder = folder if folder.is_absolute() else ROOT / folder
    if not folder.exists():
        raise FileNotFoundError(f"上传目录不存在: {folder}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    is_private = private if private is not None else bool(hcfg.get("private", True))

    try:
        api.create_repo(repo_id=repo, private=is_private, exist_ok=True)
    except Exception as e:
        console.print(f"[yellow]create_repo: {e}[/yellow]")

    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo,
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/{repo}"
    console.print(f"[green]已上传[/green] {folder} → {url}")
    return url
