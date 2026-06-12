from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config
from .data_validate import validate_jsonl
from .notify import notify

console = Console()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_inbox(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    wcfg = cfg.get("inbox") or {}
    inbox = ROOT / wcfg.get("dir", "data/inbox")
    processed = ROOT / wcfg.get("processed_dir", "data/inbox/processed")
    failed = ROOT / wcfg.get("failed_dir", "data/inbox/failed")
    train_target = ROOT / wcfg.get("merge_into", "data/train.jsonl")
    template = (cfg.get("dataset") or {}).get("chat_template", "qwen")

    inbox.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    failed.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "failed": 0, "files": []}
    for f in sorted(inbox.glob("*.jsonl")):
        report = validate_jsonl(f, template=template)
        if not report.ok:
            dest = failed / f.name
            shutil.move(str(f), dest)
            stats["failed"] += 1
            stats["files"].append({"file": f.name, "status": "failed", "errors": len(report.errors)})
            continue

        # append to train
        content = f.read_text(encoding="utf-8")
        with train_target.open("a", encoding="utf-8") as out:
            out.write(content if content.endswith("\n") else content + "\n")

        dest = processed / f"{f.stem}_{_now()[:19].replace(':', '')}.jsonl"
        shutil.move(str(f), dest)
        stats["processed"] += 1
        stats["files"].append({"file": f.name, "status": "ok"})

    if stats["processed"]:
        if bool(wcfg.get("auto_pipeline", False)):
            console.print("[cyan]触发自动 Pipeline[/cyan]")
            subprocess.Popen([sys.executable, str(ROOT / "run_pipeline.py")], cwd=ROOT)
            notify("lmstudio-finetune", f"收件箱触发 Pipeline，合并 {stats['processed']} 个文件")
        elif bool(wcfg.get("auto_alignment", False)):
            console.print("[cyan]触发自动 SFT→DPO[/cyan]")
            subprocess.Popen([sys.executable, str(ROOT / "run_alignment.py")], cwd=ROOT)
            notify("lmstudio-finetune", f"收件箱触发对齐训练，合并 {stats['processed']} 个文件")
        elif bool(wcfg.get("auto_train", False)):
            console.print("[cyan]触发自动训练[/cyan]")
            subprocess.Popen([sys.executable, str(ROOT / "train.py")], cwd=ROOT)
            notify("lmstudio-finetune", f"收件箱触发训练，合并 {stats['processed']} 个文件")

    return stats


def watch_loop(cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or load_config()
    interval = int((cfg.get("inbox") or {}).get("poll_seconds", 60))
    console.print(f"[green]监视收件箱[/green] 每 {interval}s 扫描一次 (Ctrl+C 退出)")
    try:
        while True:
            result = process_inbox(cfg)
            if result["processed"] or result["failed"]:
                console.print(json.dumps(result, ensure_ascii=False))
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("已停止监视")
