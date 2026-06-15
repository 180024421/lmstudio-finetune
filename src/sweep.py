from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config

console = Console()


def run_sweep(
    grid: dict[str, list[Any]],
    *,
    dry_run: bool = False,
    base_config: Path | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config(base_config)
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    results: list[dict[str, Any]] = []

    for combo in itertools.product(*values):
        run_cfg = copy.deepcopy(cfg)
        params = dict(zip(keys, combo))
        for dotted, val in params.items():
            parts = dotted.split(".")
            node = run_cfg
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val

        run_id = "_".join(f"{k.split('.')[-1]}{v}" for k, v in params.items())
        out_dir = ROOT / "output" / f"sweep_{run_id}"
        run_cfg["output_dir"] = str(out_dir.relative_to(ROOT)).replace("\\", "/")
        run_cfg.setdefault("lora", {})["name"] = f"sweep_{run_id}"

        sweep_cfg_path = out_dir / "sweep_config.yaml"
        if not dry_run:
            import yaml

            out_dir.mkdir(parents=True, exist_ok=True)
            sweep_cfg_path.write_text(yaml.dump(run_cfg, allow_unicode=True), encoding="utf-8")

        entry = {"params": params, "config": str(sweep_cfg_path), "output_dir": str(out_dir)}
        results.append(entry)
        console.print(f"[cyan]Sweep[/cyan] {params}")

        if not dry_run:
            subprocess.run(
                [sys.executable, str(ROOT / "train.py"), "--config", str(sweep_cfg_path)],
                cwd=ROOT,
                check=False,
            )
    return results


def save_sweep_manifest(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
