#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.import_video_promo import import_video_promo_jobs


def main() -> None:
    p = argparse.ArgumentParser(description="从 video-promo-pipeline 任务目录导入训练数据")
    p.add_argument("jobs_root", help="jobs 根目录，如 ../video-promo-pipeline/output/jobs")
    p.add_argument("-o", "--output", default="data/from_video_promo.jsonl")
    p.add_argument("--no-recursive", action="store_true")
    args = p.parse_args()
    stats = import_video_promo_jobs(
        Path(args.jobs_root),
        Path(args.output),
        recursive=not args.no_recursive,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
