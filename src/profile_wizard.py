"""dev/prod Profile 差异说明。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config_loader import ROOT, load_config


def profile_diff_markdown() -> str:
    dev_path = ROOT / "config.dev.yaml"
    prod_path = ROOT / "config.prod.yaml"
    lines = ["## Profile 对比", ""]
    for name, path in [("dev", dev_path), ("prod", prod_path)]:
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            lines.append(f"### {name}")
            lines.append("```yaml")
            lines.append(yaml.dump(data, allow_unicode=True, default_flow_style=False).strip())
            lines.append("```")
            lines.append("")
    lines.append("用法: `set LMSTUDIO_PROFILE=dev` 或 `python train.py --profile dev`")
    return "\n".join(lines)


def recommend_profile(*, vram_gb: float | None = None) -> str:
    if vram_gb is not None and vram_gb < 6:
        return "dev（小样本快速试跑）"
    return "dev 用于调试，prod 用于正式训练与回归门禁"
