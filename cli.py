#!/usr/bin/env python3
"""统一 CLI：python cli.py <子命令> [参数...]"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print(
            "子命令: train|eval|export|validate|web|api|pipeline|promo|alignment|sweep|distill|"
            "judge-filter|orpo|kto|dpo|benchmark|doctor|merge|augment|diff|shuffle|version|"
            "train-report|train-multi|crawl|upload-hub|usage|lora-ab|release|health|smoke|"
            "split|stats|hard-examples|dialogue-analytics"
        )
        raise SystemExit(1)
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    mapping = {
        "train": "train.py",
        "eval": "eval.py",
        "export": "export.py",
        "validate": "validate_data.py",
        "web": "web/app.py",
        "api": "serve_api.py",
        "pipeline": "run_pipeline.py",
        "sweep": "sweep_train.py",
        "distill": "distill_data.py",
        "judge-filter": "judge_filter.py",
        "orpo": "orpo_train.py",
        "kto": "kto_train.py",
        "dpo": "dpo_train.py",
        "benchmark": "run_benchmark.py",
        "doctor": "doctor.py",
        "merge": "merge_data.py",
        "augment": "augment_data.py",
        "diff": "diff_data.py",
        "shuffle": "shuffle_data.py",
        "version": "version_cli.py",
        "train-report": "train_report.py",
        "train-multi": "train_multi.py",
        "alignment": "run_alignment.py",
        "crawl": "crawl_docs.py",
        "upload-hub": "upload_hub.py",
        "usage": "usage_cli.py",
        "promo": "run_promo_pipeline.py",
        "lora-ab": "lora_ab_report.py",
        "release": "release_check.py",
        "health": "lmstudio_health.py",
        "smoke": "smoke_test.py",
        "split": "split_data.py",
        "stats": "stats_data.py",
        "hard-examples": "hard_example_sampler.py",
        "dialogue-analytics": "dialogue_analytics.py",
    }
    script = mapping.get(cmd)
    if not script:
        print(f"未知子命令: {cmd}")
        raise SystemExit(1)
    import runpy

    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
