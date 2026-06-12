from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or ROOT / "config.yaml"
    if not p.exists():
        p = ROOT / "config.example.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
