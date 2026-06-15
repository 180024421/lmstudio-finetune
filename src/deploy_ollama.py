from __future__ import annotations

import subprocess
from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config
from .export_model import write_ollama_modelfile

console = Console()


def deploy_to_ollama(cfg: dict[str, Any] | None = None, *, model_name: str = "") -> str:
    cfg = cfg or load_config()
    ocfg = cfg.get("ollama") or {}
    name = model_name or ocfg.get("model_name", "my-finetuned")
    merged = ROOT / (cfg.get("export") or {}).get("merged_dir", "output/merged")
    if not merged.exists():
        raise FileNotFoundError(f"合并目录不存在: {merged}，请先 export")

    modelfile = write_ollama_modelfile(merged, cfg)
    cmd = ["ollama", "create", name, "-f", str(modelfile)]
    console.print(f"[cyan]执行[/cyan] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    console.print(f"[green]Ollama 模型已创建[/green] ollama run {name}")
    return name
