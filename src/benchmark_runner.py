from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config_loader import ROOT, load_config
from .data_io import iter_jsonl
from .eval_runner import _extract_prompt_expected, _score_response, save_report
from .llm_judge import judge_response
from .lmstudio_client import chat as lm_chat
from .lmstudio_client import check_lm_studio

console = Console()
DEFAULT_BENCH = ROOT / "benchmarks" / "default.jsonl"


@dataclass
class BenchResult:
    name: str
    rule_score: float = 0.0
    judge_score: float | None = None
    samples: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rule_score": round(self.rule_score, 4),
            "judge_score": round(self.judge_score, 4) if self.judge_score is not None else None,
            "samples": self.samples,
            "details": self.details,
        }


def run_benchmark(
    bench_path: Path | None = None,
    cfg: dict[str, Any] | None = None,
    *,
    use_judge: bool = False,
    max_samples: int = 0,
) -> BenchResult:
    cfg = cfg or load_config()
    path = bench_path or DEFAULT_BENCH
    if not path.exists():
        raise FileNotFoundError(path)
    if not check_lm_studio(cfg):
        raise RuntimeError("LM Studio 未连接")

    bcfg = cfg.get("benchmark") or {}
    name = bcfg.get("name") or path.stem
    result = BenchResult(name=name)
    rule_scores: list[float] = []
    judge_scores: list[float] = []

    for _, row in iter_jsonl(path):
        if max_samples and result.samples >= max_samples:
            break
        prompt, expected = _extract_prompt_expected(row)
        actual = lm_chat(prompt, cfg)
        rs, rnote = _score_response(expected, actual, row.get("keywords"))
        rule_scores.append(rs)
        detail: dict[str, Any] = {"prompt": prompt[:80], "rule": rs, "rule_note": rnote}
        if use_judge:
            js, jnote = judge_response(prompt, expected, actual, cfg)
            judge_scores.append(js)
            detail["judge"] = js
            detail["judge_note"] = jnote
        result.details.append(detail)
        result.samples += 1

    result.rule_score = sum(rule_scores) / len(rule_scores) if rule_scores else 0.0
    if judge_scores:
        result.judge_score = sum(judge_scores) / len(judge_scores)
    return result


def print_benchmark(result: BenchResult) -> None:
    table = Table(title=f"Benchmark: {result.name}")
    table.add_column("规则分")
    table.add_column("评审分")
    table.add_column("样本")
    table.add_row(
        f"{result.rule_score:.3f}",
        f"{result.judge_score:.3f}" if result.judge_score is not None else "-",
        str(result.samples),
    )
    console.print(table)


def run_and_save(
    bench_path: Path | None = None,
    cfg: dict[str, Any] | None = None,
    *,
    use_judge: bool = False,
    output: Path | None = None,
) -> BenchResult:
    result = run_benchmark(bench_path, cfg, use_judge=use_judge)
    out = output or ROOT / "output" / "benchmark_report.json"
    save_report(result.to_dict(), out)
    return result
