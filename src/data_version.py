from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import ROOT
from .data_io import load_rows

VERSIONS = ROOT / "data" / "versions.json"


def dataset_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    rows = load_rows(path)
    for row in rows:
        h.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()[:16]


def register_dataset_version(path: Path, *, note: str = "") -> dict[str, Any]:
    path = path.resolve()
    try:
        rel = str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        rel = str(path)
    entry = {
        "path": rel,
        "count": len(load_rows(path)),
        "fingerprint": dataset_fingerprint(path),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    versions: list[dict[str, Any]] = []
    if VERSIONS.exists():
        versions = json.loads(VERSIONS.read_text(encoding="utf-8"))
    versions.append(entry)
    VERSIONS.parent.mkdir(parents=True, exist_ok=True)
    VERSIONS.write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def list_versions(*, path: str | None = None) -> list[dict[str, Any]]:
    if not VERSIONS.exists():
        return []
    versions: list[dict[str, Any]] = json.loads(VERSIONS.read_text(encoding="utf-8"))
    if path:
        return [v for v in versions if v.get("path") == path or v.get("path", "").endswith(path)]
    return versions


def versions_markdown() -> str:
    versions = list_versions()
    if not versions:
        return "*(暂无登记的数据集版本)*"
    lines = ["| 时间 | 路径 | 条数 | 指纹 | 备注 |", "|------|------|------|------|------|"]
    for v in reversed(versions[-30:]):
        lines.append(
            f"| {v.get('registered_at', '')[:19]} | {v.get('path', '')} | {v.get('count', '')} "
            f"| `{v.get('fingerprint', '')}` | {v.get('note', '')} |"
        )
    return "\n".join(lines)
