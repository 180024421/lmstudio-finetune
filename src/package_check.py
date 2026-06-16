"""包安装状态：缺失 vs 损坏。"""

from __future__ import annotations

import importlib.util
from typing import Any


def check_package(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"name": name, "status": "missing", "fix": f"pip install {name}"}
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "ok")
        return {"name": name, "status": "ok", "version": str(ver)}
    except Exception as e:
        return {
            "name": name,
            "status": "corrupted",
            "error": str(e),
            "fix": f"pip install --force-reinstall {name}",
        }
