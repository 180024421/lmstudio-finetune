from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_config_path(path: Path | None = None, profile: str | None = None) -> Path:
    if path:
        return path
    prof = profile or os.environ.get("LMSTUDIO_PROFILE", "").strip()
    if prof:
        prof_path = ROOT / f"config.{prof}.yaml"
        if prof_path.exists():
            return prof_path
    main = ROOT / "config.yaml"
    if main.exists():
        return main
    return ROOT / "config.example.yaml"


def load_config(path: Path | None = None, profile: str | None = None) -> dict[str, Any]:
    """加载配置；若指定 profile，在基础 config 上深度合并 config.{profile}.yaml。"""
    base_path = ROOT / "config.yaml"
    if not base_path.exists():
        base_path = ROOT / "config.example.yaml"
    if path and path.exists():
        base_path = path

    cfg: dict[str, Any] = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

    prof = profile or os.environ.get("LMSTUDIO_PROFILE", "").strip()
    if prof:
        prof_path = ROOT / f"config.{prof}.yaml"
        if prof_path.exists():
            prof_cfg = yaml.safe_load(prof_path.read_text(encoding="utf-8")) or {}
            cfg = _deep_merge(cfg, prof_cfg)
        elif profile:
            raise FileNotFoundError(f"配置 profile 不存在: {prof_path}")

    return cfg
