#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import ROOT, load_config
from src.sweep import run_sweep, save_sweep_manifest

DEFAULT_GRID = {
    "lora.r": [8, 16],
    "train.learning_rate": [1e-4, 2e-4],
}


def main() -> None:
    p = argparse.ArgumentParser(description="超参网格搜索")
    p.add_argument("--grid", default=None, help="JSON 网格，默认 lora.r + lr")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    grid = json.loads(args.grid) if args.grid else DEFAULT_GRID
    cfg_path = Path(args.config) if args.config else None
    load_config(cfg_path)
    results = run_sweep(grid, dry_run=args.dry_run, base_config=cfg_path)
    manifest = ROOT / "output" / "sweep_manifest.json"
    save_sweep_manifest(results, manifest)
    print(f"共 {len(results)} 组 → {manifest}")


if __name__ == "__main__":
    main()
