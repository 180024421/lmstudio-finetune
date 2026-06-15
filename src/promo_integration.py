from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config
from .data_io import write_jsonl
from .data_validate import validate_jsonl
from .import_video_promo import import_job_dir

console = Console()

BRIDGE_FILENAME = "finetune_bridge.json"
DEFAULT_PROMO_REL = "../video-promo-pipeline"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_promo_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"video-promo 目录不存在: {p}")

    cfg = load_config()
    pcfg = cfg.get("promo") or {}
    configured = pcfg.get("pipeline_root") or ""
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p.resolve()

    for candidate in (
        ROOT / DEFAULT_PROMO_REL,
        ROOT.parent / "video-promo-pipeline",
        Path(r"E:\xiangmu\video-promo-pipeline"),
    ):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("未找到 video-promo-pipeline，请设置 promo.pipeline_root")


def bridge_path(promo_root: Path) -> Path:
    return promo_root / "data" / BRIDGE_FILENAME


def discover_job_dirs(jobs_root: Path, *, recursive: bool = True) -> list[Path]:
    pattern = "**/promo_copy.json" if recursive else "*/promo_copy.json"
    jobs: list[Path] = []
    seen: set[str] = set()
    for promo_file in sorted(jobs_root.glob(pattern)):
        job_dir = promo_file.parent
        key = str(job_dir.resolve())
        if key not in seen:
            seen.add(key)
            jobs.append(job_dir)
    return jobs


def import_promo_with_job_split(
    jobs_root: Path,
    output_dir: Path,
    *,
    eval_job_ratio: float = 0.2,
    seed: int = 42,
    recursive: bool = True,
) -> dict[str, Any]:
    """按 job 切分训练/评测集，避免同一视频文案泄漏到 eval。"""
    job_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for job_dir in discover_job_dirs(jobs_root, recursive=recursive):
        rows = import_job_dir(job_dir)
        if rows:
            job_rows.append((job_dir.name, rows))

    if not job_rows:
        raise ValueError(f"未在 {jobs_root} 找到可导入的 promo 任务")

    rng = random.Random(seed)
    rng.shuffle(job_rows)

    n_eval_jobs = max(1, int(len(job_rows) * eval_job_ratio)) if len(job_rows) > 1 else 0
    if len(job_rows) == 1:
        n_eval_jobs = 0

    eval_jobs = job_rows[:n_eval_jobs]
    train_jobs = job_rows[n_eval_jobs:]

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for name, rows in train_jobs:
        for r in rows:
            train_rows.append({**r, "_job": name})
    for name, rows in eval_jobs:
        for r in rows:
            eval_rows.append({**r, "_job": name})

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "promo_train.jsonl"
    eval_path = output_dir / "promo_eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    return {
        "jobs_total": len(job_rows),
        "train_jobs": [n for n, _ in train_jobs],
        "eval_jobs": [n for n, _ in eval_jobs],
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
    }


def write_bridge(promo_root: Path, bridge: dict[str, Any]) -> Path:
    path = bridge_path(promo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": _now(), **bridge}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]已回写 bridge[/green] {path}")
    return path


def apply_bridge_to_promo_config(promo_root: Path, bridge: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """将推荐模型写入 video-promo 的 config.yaml（若存在）。"""
    cfg_path = promo_root / "config.yaml"
    if not cfg_path.exists():
        return {"ok": False, "reason": "config.yaml 不存在，仅写入 bridge 文件"}

    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    model = bridge.get("recommended_lm_studio_model") or bridge.get("recommended_model") or ""
    if not model:
        return {"ok": False, "reason": "bridge 中无推荐模型"}

    lm = dict(cfg.get("lm_studio") or {})
    old = lm.get("model", "")
    lm["model"] = model
    cfg["lm_studio"] = lm

    finetune = dict(cfg.get("finetune") or {})
    finetune["last_bridge_at"] = bridge.get("updated_at")
    finetune["recommended_lora"] = bridge.get("recommended_lora", "")
    finetune["eval_score"] = bridge.get("eval_score")
    cfg["finetune"] = finetune

    if not dry_run:
        cfg_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    return {"ok": True, "config": str(cfg_path), "old_model": old, "new_model": model}


def run_promo_cycle(
    cfg: dict[str, Any] | None = None,
    *,
    promo_root: Path | None = None,
    jobs_root: Path | None = None,
    skip_train: bool = False,
    skip_export: bool = False,
    skip_eval: bool = False,
    skip_ab: bool = False,
    apply_config: bool = False,
) -> dict[str, Any]:
    """video-promo 深度联动：导入 → 训练 → 评测 → A/B → 回写 bridge。"""
    cfg = cfg or load_config()
    pcfg = cfg.get("promo") or {}

    promo = promo_root or resolve_promo_root(pcfg.get("pipeline_root"))
    jobs = jobs_root or Path(pcfg.get("jobs_dir") or promo / "output")
    if not jobs.is_absolute():
        jobs = ROOT / jobs if (ROOT / jobs).exists() else promo / "output" if (promo / "output").exists() else jobs

    if not Path(jobs).exists():
        raise FileNotFoundError(f"jobs 目录不存在: {jobs}")

    data_dir = ROOT / pcfg.get("data_dir", "data/promo")
    split = import_promo_with_job_split(
        Path(jobs),
        data_dir,
        eval_job_ratio=float(pcfg.get("eval_job_ratio", 0.2)),
        seed=int(pcfg.get("seed", 42)),
    )

    train_file = Path(split["train_file"])
    eval_file = Path(split["eval_file"])
    template = (cfg.get("dataset") or {}).get("chat_template", "qwen")

    vr = validate_jsonl(train_file, template=template)
    if not vr.ok:
        raise ValueError(f"训练集校验失败: {vr.errors[0].message}")

    result: dict[str, Any] = {"import": split, "promo_root": str(promo)}

    run_cfg = dict(cfg)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    run_cfg["dataset"] = {
        **(cfg.get("dataset") or {}),
        "train_file": _rel(train_file),
        "eval_file": _rel(eval_file),
    }
    lora = dict(run_cfg.get("lora") or {})
    lora.setdefault("tags", [])
    if "promo" not in lora["tags"]:
        lora["tags"] = list(lora["tags"]) + ["promo"]
    run_cfg["lora"] = lora

    adapter_path: Path | None = None
    if not skip_train:
        from .train_lora import run_train

        adapter_path = run_train(run_cfg)
        result["adapter"] = str(adapter_path)

    if not skip_export:
        from .export_model import export_all

        result["export"] = {k: str(v) for k, v in (export_all(run_cfg) or {}).items() if v}

    eval_score: float | None = None
    if not skip_eval and eval_file.exists():
        from .eval_runner import run_eval_judge, run_eval_lmstudio

        use_judge = bool(pcfg.get("use_judge", True))
        max_n = int(pcfg.get("eval_samples", 20))
        if use_judge:
            er = run_eval_judge(eval_file, run_cfg, max_samples=max_n)
        else:
            er = run_eval_lmstudio(eval_file, run_cfg, max_samples=max_n)
        eval_score = er.avg_score
        result["eval_score"] = eval_score
        result["eval_mode"] = er.mode

        from .eval_runner import save_report, save_report_html

        report_dir = ROOT / "output" / "promo_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        save_report(er, report_dir / "latest_eval.json")
        save_report_html(er, report_dir / "latest_eval.html")
        result["eval_report_html"] = str(report_dir / "latest_eval.html")

    ab_result: dict[str, Any] | None = None
    if not skip_ab:
        from .lora_ab_report import compare_loras, save_lora_ab_html

        names = pcfg.get("ab_loras") or []
        if not names:
            tag = pcfg.get("ab_tag", "promo")
            from .lora_manager import list_adapters

            names = [a["name"] for a in list_adapters() if tag in (a.get("tags") or [])]
        if len(names) >= 2 and eval_file.exists():
            ab_result = compare_loras(
                names,
                eval_file,
                run_cfg,
                max_samples=int(pcfg.get("ab_samples", 15)),
                mode="judge" if pcfg.get("use_judge", True) else "lmstudio",
            )
            ab_html = ROOT / "output" / "promo_reports" / "lora_ab_report.html"
            save_lora_ab_html(ab_result, ab_html)
            result["ab_report"] = ab_result
            result["ab_report_html"] = str(ab_html)

    recommended_lora = ""
    recommended_model = pcfg.get("lm_studio_model") or (run_cfg.get("lm_studio") or {}).get("model") or ""
    if ab_result and ab_result.get("winner"):
        recommended_lora = ab_result["winner"]
        winner_entry = next(
            (m for m in ab_result.get("ranking", []) if m.get("name") == recommended_lora),
            {},
        )
        recommended_model = winner_entry.get("lm_studio_model") or recommended_model
        eval_score = winner_entry.get("avg_score", eval_score)
    elif not skip_train:
        recommended_lora = (run_cfg.get("lora") or {}).get("name", "")

    bridge = {
        "source": "lmstudio-finetune",
        "recommended_lora": recommended_lora,
        "recommended_lm_studio_model": recommended_model,
        "eval_score": eval_score,
        "train_file": split["train_file"],
        "eval_file": split["eval_file"],
        "train_samples": split["train_samples"],
        "eval_samples": split["eval_samples"],
        "ab_winner": ab_result.get("winner") if ab_result else None,
        "ab_ranking": ab_result.get("ranking") if ab_result else None,
        "reports": {
            "eval_html": result.get("eval_report_html"),
            "ab_html": result.get("ab_report_html"),
        },
        "export": result.get("export"),
        "adapter": result.get("adapter"),
    }
    bridge_path_written = write_bridge(promo, bridge)
    result["bridge"] = str(bridge_path_written)

    if apply_config or bool(pcfg.get("auto_apply_config", False)):
        result["config_patch"] = apply_bridge_to_promo_config(promo, bridge)

    from .experiment_registry import register_experiment

    register_experiment(
        name=f"promo-{(run_cfg.get('lora') or {}).get('name', 'run')}",
        cfg=run_cfg,
        eval_score=eval_score,
        eval_mode=result.get("eval_mode", ""),
        notes=f"video-promo jobs={split['jobs_total']}",
    )

    console.print("[bold green]video-promo 联动周期完成[/bold green]")
    return result
