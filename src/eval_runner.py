from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config_loader import ROOT, load_config
from .data_io import iter_jsonl, load_rows
from .llm_judge import judge_response
from .lmstudio_client import chat as lm_chat
from .lmstudio_client import check_lm_studio

console = Console()


@dataclass
class EvalSample:
    prompt: str
    expected: str
    actual: str = ""
    score: float = 0.0
    notes: str = ""


@dataclass
class EvalReport:
    mode: str
    total: int = 0
    avg_score: float = 0.0
    samples: list[EvalSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total": self.total,
            "avg_score": round(self.avg_score, 4),
            "samples": [
                {"prompt": s.prompt, "expected": s.expected, "actual": s.actual, "score": s.score, "notes": s.notes}
                for s in self.samples
            ],
        }


def _extract_prompt_expected(row: dict[str, Any]) -> tuple[str, str]:
    if "messages" in row:
        msgs = row["messages"]
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        expected = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        return user, expected
    if "instruction" in row:
        inst = row.get("instruction", "")
        inp = row.get("input", "")
        user = f"{inst}\n{inp}".strip() if inp else inst
        return user, row.get("output", "")
    if "prompt" in row and "chosen" in row:
        return row["prompt"], row["chosen"]
    raise ValueError("无法解析评测样本")


def _score_response(expected: str, actual: str, keywords: list[str] | None = None) -> tuple[float, str]:
    if not actual.strip():
        return 0.0, "空回复"
    exp = expected.strip().lower()
    act = actual.strip().lower()
    if exp == act:
        return 1.0, "完全匹配"
    # 子串包含
    if exp in act or act in exp:
        return 0.85, "部分包含"
    # 关键词命中率
    if keywords:
        hits = sum(1 for k in keywords if k.lower() in act)
        if hits:
            return min(0.9, hits / len(keywords)), f"关键词命中 {hits}/{len(keywords)}"
    # 字符 bigram 重叠
    def bigrams(s: str) -> set[str]:
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()

    eb, ab = bigrams(exp), bigrams(act)
    if not eb:
        return 0.3, "期望为空"
    overlap = len(eb & ab) / len(eb)
    return round(overlap * 0.8, 3), f"bigram 重叠 {overlap:.2f}"


def run_eval_lmstudio(
    eval_path: Path,
    cfg: dict[str, Any] | None = None,
    *,
    max_samples: int = 20,
    model_override: str | None = None,
) -> EvalReport:
    cfg = cfg or load_config()
    if not check_lm_studio(cfg):
        raise RuntimeError("LM Studio 未连接，请先启动 Local Server")

    lcfg = dict(cfg.get("lm_studio") or {})
    if model_override:
        lcfg["model"] = model_override
    cfg = {**cfg, "lm_studio": lcfg}

    report = EvalReport(mode="lm_studio")
    scores: list[float] = []

    for lineno, row in iter_jsonl(eval_path):
        if max_samples and report.total >= max_samples:
            break
        prompt, expected = _extract_prompt_expected(row)
        keywords = row.get("keywords") or []
        try:
            actual = lm_chat(prompt, cfg)
        except Exception as e:
            actual = ""
            score, notes = 0.0, str(e)
        else:
            score, notes = _score_response(expected, actual, keywords)

        report.samples.append(EvalSample(prompt, expected, actual, score, notes))
        scores.append(score)
        report.total += 1

    report.avg_score = sum(scores) / len(scores) if scores else 0.0
    return report


def run_eval_local(
    eval_path: Path,
    cfg: dict[str, Any] | None = None,
    *,
    adapter_dir: Path | None = None,
    max_samples: int = 20,
) -> EvalReport:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or load_config()
    base_model = cfg["base_model"]
    adapter_dir = adapter_dir or (ROOT / cfg.get("output_dir", "output/run1") / "adapter")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter_dir.exists():
        model = PeftModel.from_pretrained(model, adapter_dir)

    report = EvalReport(mode="local_hf")
    scores: list[float] = []

    for _, row in iter_jsonl(eval_path):
        if max_samples and report.total >= max_samples:
            break
        prompt, expected = _extract_prompt_expected(row)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        actual = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        score, notes = _score_response(expected, actual, row.get("keywords"))
        report.samples.append(EvalSample(prompt, expected, actual, score, notes))
        scores.append(score)
        report.total += 1

    report.avg_score = sum(scores) / len(scores) if scores else 0.0
    return report


def run_eval_judge(
    eval_path: Path,
    cfg: dict[str, Any] | None = None,
    *,
    max_samples: int = 10,
    model_override: str | None = None,
) -> EvalReport:
    cfg = cfg or load_config()
    if not check_lm_studio(cfg):
        raise RuntimeError("LM Studio 未连接")

    if model_override:
        lcfg = dict(cfg.get("lm_studio") or {})
        lcfg["model"] = model_override
        cfg = {**cfg, "lm_studio": lcfg}

    report = EvalReport(mode="llm_judge")
    scores: list[float] = []

    for _, row in iter_jsonl(eval_path):
        if max_samples and report.total >= max_samples:
            break
        prompt, expected = _extract_prompt_expected(row)
        try:
            actual = lm_chat(prompt, cfg)
            score, notes = judge_response(prompt, expected, actual, cfg)
        except Exception as e:
            actual, score, notes = "", 0.0, str(e)
        report.samples.append(EvalSample(prompt, expected, actual, score, notes))
        scores.append(score)
        report.total += 1

    report.avg_score = sum(scores) / len(scores) if scores else 0.0
    return report


def compare_models(
    eval_path: Path,
    cfg: dict[str, Any] | None = None,
    *,
    base_model_name: str,
    finetuned_model_name: str,
    max_samples: int = 10,
) -> dict[str, Any]:
    """在 LM Studio 中切换两个已加载模型名进行对比（需分别加载或手动切换）。"""
    base_report = run_eval_lmstudio(eval_path, cfg, max_samples=max_samples, model_override=base_model_name)
    ft_report = run_eval_lmstudio(eval_path, cfg, max_samples=max_samples, model_override=finetuned_model_name)
    return {
        "base": base_report.to_dict(),
        "finetuned": ft_report.to_dict(),
        "delta": round(ft_report.avg_score - base_report.avg_score, 4),
    }


def print_report(report: EvalReport) -> None:
    table = Table(title=f"评测报告 ({report.mode})")
    table.add_column("分数", justify="right")
    table.add_column("问题")
    table.add_column("说明")
    for s in report.samples[:10]:
        table.add_row(f"{s.score:.2f}", s.prompt[:40], s.notes)
    console.print(table)
    console.print(f"[bold]平均得分: {report.avg_score:.3f}[/bold] ({report.total} 条)")


def save_report(report: EvalReport | dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report.to_dict() if isinstance(report, EvalReport) else report
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_report_html(report: EvalReport | dict[str, Any], path: Path) -> None:
    data = report.to_dict() if isinstance(report, EvalReport) else report
    rows = []
    for s in data.get("samples", []):
        rows.append(
            f"<tr><td>{s.get('score', 0):.2f}</td>"
            f"<td>{_html_escape(s.get('prompt', '')[:120])}</td>"
            f"<td>{_html_escape(s.get('notes', ''))}</td></tr>"
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>评测报告</title>
<style>table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px}}</style>
</head><body>
<h1>评测报告 ({_html_escape(data.get('mode', ''))})</h1>
<p>平均得分: <strong>{data.get('avg_score', 0):.3f}</strong> / {data.get('total', 0)} 条</p>
<table><tr><th>分数</th><th>问题</th><th>说明</th></tr>
{''.join(rows)}
</table></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
