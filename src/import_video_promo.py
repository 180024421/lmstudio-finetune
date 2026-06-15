from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_io import write_jsonl


def _rows_from_promo_copy(promo: dict[str, Any], transcript: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    topic = (transcript[:200] + "...") if len(transcript) > 200 else transcript

    bilibili = promo.get("bilibili")
    if isinstance(bilibili, dict):
        titles = bilibili.get("titles") or []
        desc = bilibili.get("description") or bilibili.get("recommended_title") or ""
        for title in titles:
            if not str(title).strip():
                continue
            rows.append(
                {
                    "instruction": "为以下视频内容写一条 B 站推广标题",
                    "input": topic or "（见转写稿）",
                    "output": str(title).strip(),
                }
            )
        if desc and titles:
            rows.append(
                {
                    "instruction": "为以下视频写 B 站简介",
                    "input": f"标题参考：{titles[0]}\n内容摘要：{topic}",
                    "output": str(desc).strip(),
                }
            )

    xhs = promo.get("xiaohongshu")
    if isinstance(xhs, dict):
        title = xhs.get("title", "")
        body = xhs.get("body", "")
        if title and body:
            rows.append(
                {
                    "instruction": "根据视频内容写小红书标题与正文",
                    "input": topic,
                    "output": f"{title}\n\n{body}".strip(),
                }
            )

    douyin = promo.get("douyin")
    if isinstance(douyin, dict):
        for field in ("title", "caption", "description"):
            val = douyin.get(field)
            if val:
                rows.append(
                    {
                        "instruction": f"为短视频写抖音{field}",
                        "input": topic,
                        "output": str(val).strip(),
                    }
                )

    hooks = promo.get("short_hooks")
    if isinstance(hooks, list):
        for h in hooks:
            if h:
                rows.append(
                    {
                        "instruction": "写一条短视频开头钩子",
                        "input": topic,
                        "output": str(h).strip(),
                    }
                )
    return rows


def import_job_dir(job_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    transcript = ""
    tx = job_dir / "transcript.txt"
    if tx.exists():
        transcript = tx.read_text(encoding="utf-8", errors="ignore")

    promo_path = job_dir / "promo_copy.json"
    if promo_path.exists():
        promo = json.loads(promo_path.read_text(encoding="utf-8"))
        rows.extend(_rows_from_promo_copy(promo, transcript))

    narr_path = job_dir / "narration.json"
    if narr_path.exists():
        narr = json.loads(narr_path.read_text(encoding="utf-8"))
        script = narr.get("script") or narr.get("full_text") or ""
        if script and transcript:
            rows.append(
                {
                    "instruction": "根据视频转写稿生成口语化解说词",
                    "input": transcript[:3000],
                    "output": str(script)[:4000],
                }
            )
    return rows


def import_video_promo_jobs(
    jobs_root: Path,
    output_path: Path,
    *,
    recursive: bool = True,
) -> dict[str, int]:
    all_rows: list[dict[str, Any]] = []
    job_count = 0
    pattern = "**/promo_copy.json" if recursive else "*/promo_copy.json"
    for promo_file in jobs_root.glob(pattern):
        job_dir = promo_file.parent
        rows = import_job_dir(job_dir)
        if rows:
            job_count += 1
            all_rows.extend(rows)

    write_jsonl(output_path, all_rows)
    return {"jobs": job_count, "samples": len(all_rows), "output": str(output_path)}
