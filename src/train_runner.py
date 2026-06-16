"""训练子进程管理与日志 tail。"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .config_loader import ROOT

_LOG_PATH = ROOT / "output" / "train_web.log"
_proc: subprocess.Popen | None = None


def _reader(pipe, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8", errors="replace") as f:
        for line in iter(pipe.readline, ""):
            f.write(line)
            f.flush()
        pipe.close()


def start_training(script: str = "train.py", *, extra_args: list[str] | None = None) -> dict[str, Any]:
    global _proc
    if _proc and _proc.poll() is None:
        return {"ok": False, "detail": "训练已在运行中", "pid": _proc.pid}

    _LOG_PATH.write_text("", encoding="utf-8")
    cmd = [sys.executable, str(ROOT / script)] + (extra_args or [])
    _proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert _proc.stdout is not None
    threading.Thread(target=_reader, args=(_proc.stdout, _LOG_PATH), daemon=True).start()
    return {"ok": True, "pid": _proc.pid, "cmd": " ".join(cmd), "log": str(_LOG_PATH)}


def tail_training_log(lines: int = 80) -> str:
    if not _LOG_PATH.exists():
        return "(暂无日志)"
    content = _LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def training_status() -> dict[str, Any]:
    global _proc
    running = _proc is not None and _proc.poll() is None
    return {
        "running": running,
        "pid": _proc.pid if _proc else None,
        "exit_code": _proc.poll() if _proc else None,
        "log_path": str(_LOG_PATH),
    }


def stop_training() -> str:
    global _proc
    if not _proc or _proc.poll() is not None:
        return "无运行中的训练进程"
    _proc.terminate()
    try:
        _proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _proc.kill()
    _proc = None
    return "已发送停止信号"
