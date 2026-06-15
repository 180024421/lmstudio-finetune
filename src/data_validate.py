from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data_io import iter_jsonl, row_fingerprint
from .dataset import row_to_training_text


@dataclass
class ValidationIssue:
    line: int
    level: str  # error | warning
    message: str


@dataclass
class ValidationReport:
    path: Path
    total: int = 0
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    duplicates: int = 0

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "total": self.total,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "duplicates": self.duplicates,
            "ok": self.ok,
        }


def validate_jsonl(
    path: Path,
    *,
    template: str = "qwen",
    max_chars: int = 32000,
    require_assistant: bool = True,
) -> ValidationReport:
    report = ValidationReport(path=path)
    seen: set[str] = set()

    if not path.exists():
        report.errors.append(ValidationIssue(0, "error", f"文件不存在: {path}"))
        return report

    for lineno, row in iter_jsonl(path):
        report.total += 1
        fp = row_fingerprint(row)
        if fp in seen:
            report.duplicates += 1
            report.warnings.append(ValidationIssue(lineno, "warning", "重复样本"))
        seen.add(fp)

        if "messages" in row:
            msgs = row["messages"]
            if not isinstance(msgs, list) or not msgs:
                report.errors.append(ValidationIssue(lineno, "error", "messages 必须为非空列表"))
                continue
            has_assistant = any(
                m.get("role") == "assistant" and str(m.get("content", "")).strip() for m in msgs if isinstance(m, dict)
            )
            if require_assistant and not has_assistant:
                report.errors.append(ValidationIssue(lineno, "error", "缺少 assistant 回复"))
            for i, m in enumerate(msgs):
                if not isinstance(m, dict):
                    report.errors.append(ValidationIssue(lineno, "error", f"messages[{i}] 不是对象"))
                elif not str(m.get("content", "")).strip():
                    report.warnings.append(ValidationIssue(lineno, "warning", f"messages[{i}] content 为空"))
        elif "instruction" in row:
            if not str(row.get("output", "")).strip() and require_assistant:
                report.errors.append(ValidationIssue(lineno, "error", "instruction 格式缺少 output"))
        elif row.get("context") and (row.get("question") or row.get("instruction")):
            if not str(row.get("answer") or row.get("output", "")).strip() and require_assistant:
                report.errors.append(ValidationIssue(lineno, "error", "RAG 格式缺少 answer"))
        else:
            report.errors.append(ValidationIssue(lineno, "error", "需要 messages 或 instruction 字段"))
            continue

        try:
            text = row_to_training_text(row, template)["text"]
            if len(text) > max_chars:
                report.warnings.append(
                    ValidationIssue(lineno, "warning", f"文本过长 ({len(text)} chars)，可能超 max_seq_length")
                )
        except Exception as e:
            report.errors.append(ValidationIssue(lineno, "error", str(e)))

    return report
