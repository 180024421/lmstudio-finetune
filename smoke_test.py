#!/usr/bin/env python3
"""E2E 冒烟：依赖导入 + dry-run + Web 模块加载。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    errors: list[str] = []

    for mod in ("torch", "transformers", "gradio", "peft"):
        try:
            __import__(mod)
        except ImportError as e:
            errors.append(f"import {mod}: {e}")

    for script, args in [("train.py", ["--dry-run"]), ("validate_config.py", [])]:
        r = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            errors.append(f"{script} exit {r.returncode}: {(r.stderr or r.stdout)[:200]}")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("webapp", ROOT / "web" / "app.py")
        assert spec and spec.loader
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    except Exception as e:
        errors.append(f"web/app.py load: {e}")

    if errors:
        print("SMOKE FAIL")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
