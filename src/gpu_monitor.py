"""训练时 GPU 显存采样。"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import ROOT


def sample_gpu_stats() -> dict[str, Any]:
    entry: dict[str, Any] = {"ts": datetime.now(timezone.utc).isoformat()}
    try:
        import torch

        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            entry.update(
                {
                    "device": torch.cuda.get_device_name(i),
                    "memory_allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                    "memory_reserved_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2),
                }
            )
            return entry
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            entry.update({"memory_used_mb": parts[0], "memory_total_mb": parts[1], "gpu_util_pct": parts[2]})
    except Exception as e:
        entry["error"] = str(e)
    return entry


def append_gpu_metric(run_dir: Path | None = None) -> Path:
    run_dir = run_dir or ROOT / "output" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "gpu_metrics.jsonl"
    stat = sample_gpu_stats()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stat, ensure_ascii=False) + "\n")
    return path
