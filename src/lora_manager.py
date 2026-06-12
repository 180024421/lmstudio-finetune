from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config_loader import ROOT


REGISTRY_PATH = ROOT / "output" / "lora_registry.yaml"


def _rel_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p.resolve())


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"adapters": []}
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {"adapters": []}


def save_registry(data: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def register_adapter(
    name: str,
    adapter_dir: Path,
    *,
    base_model: str,
    description: str = "",
    tags: list[str] | None = None,
    lm_studio_model: str = "",
    eval_score: float | None = None,
    gguf_path: str = "",
) -> dict[str, Any]:
    adapter_dir = adapter_dir.resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)

    reg = load_registry()
    entry = {
        "name": name,
        "path": _rel_path(adapter_dir),
        "base_model": base_model,
        "description": description,
        "tags": tags or [],
        "created_at": _now(),
        "lm_studio_model": lm_studio_model,
        "eval_score": eval_score,
        "gguf_path": gguf_path,
    }
    reg["adapters"] = [a for a in reg.get("adapters", []) if a.get("name") != name]
    reg["adapters"].append(entry)
    save_registry(reg)
    return entry


def list_adapters() -> list[dict[str, Any]]:
    return load_registry().get("adapters", [])


def resolve_adapter(name: str) -> Path:
    for a in list_adapters():
        if a.get("name") == name:
            p = Path(a["path"])
            return p if p.is_absolute() else ROOT / p
    raise KeyError(f"未找到 LoRA: {name}")


def export_adapter_copy(name: str, dest: Path) -> Path:
    src = resolve_adapter(name)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest


def update_adapter_metadata(name: str, **fields: Any) -> dict[str, Any]:
    reg = load_registry()
    updated: dict[str, Any] | None = None
    for a in reg.get("adapters", []):
        if a.get("name") == name:
            for k, v in fields.items():
                if v is not None:
                    a[k] = v
            updated = a
            break
    if not updated:
        raise KeyError(f"未找到 LoRA: {name}")
    save_registry(reg)
    return updated


def list_by_tag(tag: str) -> list[dict[str, Any]]:
    return [a for a in list_adapters() if tag in (a.get("tags") or [])]


def registry_to_markdown() -> str:
    lines = ["# LoRA 注册表\n"]
    for a in list_adapters():
        lines.append(f"## {a.get('name')}")
        lines.append(f"- 路径: `{a.get('path')}`")
        lines.append(f"- 基座: {a.get('base_model')}")
        lines.append(f"- 说明: {a.get('description', '')}")
        lines.append(f"- 标签: {', '.join(a.get('tags') or [])}")
        if a.get("lm_studio_model"):
            lines.append(f"- LM Studio 模型: {a.get('lm_studio_model')}")
        if a.get("eval_score") is not None:
            lines.append(f"- 评测分: {a.get('eval_score')}")
        if a.get("gguf_path"):
            lines.append(f"- GGUF: `{a.get('gguf_path')}`")
        lines.append(f"- 创建: {a.get('created_at')}\n")
    return "\n".join(lines)
