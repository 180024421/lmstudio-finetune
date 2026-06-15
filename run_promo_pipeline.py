#!/usr/bin/env python3
"""video-promo 深度联动：导入 → 训练 → 评测 → A/B → 回写 bridge。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.promo_integration import import_promo_with_job_split, resolve_promo_root, run_promo_cycle


def main() -> None:
    p = argparse.ArgumentParser(description="video-promo 深度联动流水线")
    p.add_argument("--config", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--promo-root", default=None, help="video-promo-pipeline 根目录")
    p.add_argument("--jobs", default=None, help="jobs 目录，默认 promo/output")
    p.add_argument("--import-only", action="store_true", help="仅导入并切分数据")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-ab", action="store_true")
    p.add_argument("--apply-config", action="store_true", help="回写 video-promo config.yaml 的 lm_studio.model")
    args = p.parse_args()

    cfg = load_config(Path(args.config) if args.config else None, profile=args.profile)
    promo = resolve_promo_root(args.promo_root) if args.promo_root else None

    if args.import_only:
        promo = promo or resolve_promo_root((cfg.get("promo") or {}).get("pipeline_root"))
        pcfg = cfg.get("promo") or {}
        jobs = Path(args.jobs) if args.jobs else promo / "output"
        data_dir = Path(pcfg.get("data_dir", "data/promo"))
        if not data_dir.is_absolute():
            from src.config_loader import ROOT

            data_dir = ROOT / data_dir
        stats = import_promo_with_job_split(
            jobs,
            data_dir,
            eval_job_ratio=float(pcfg.get("eval_job_ratio", 0.2)),
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    result = run_promo_cycle(
        cfg,
        promo_root=promo,
        jobs_root=Path(args.jobs) if args.jobs else None,
        skip_train=args.skip_train,
        skip_export=args.skip_export,
        skip_eval=args.skip_eval,
        skip_ab=args.skip_ab,
        apply_config=args.apply_config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
