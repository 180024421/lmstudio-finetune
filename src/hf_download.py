from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def apply_hf_mirror(mirror: str = "https://hf-mirror.com") -> None:
    os.environ.setdefault("HF_ENDPOINT", mirror)
    console.print(f"[cyan]HF_ENDPOINT[/cyan] = {os.environ['HF_ENDPOINT']}")


def download_model(model_id: str, cfg: dict[str, Any] | None = None, *, cache_dir: str | None = None) -> Path:
    cfg = cfg or {}
    hfcfg = cfg.get("huggingface") or {}
    if hfcfg.get("mirror", True):
        apply_hf_mirror(hfcfg.get("endpoint", "https://hf-mirror.com"))

    from huggingface_hub import snapshot_download

    local_dir = cache_dir or hfcfg.get("cache_dir") or None
    console.print(f"[cyan]下载模型[/cyan] {model_id}")
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    console.print(f"[green]完成[/green] {path}")
    return Path(path)
