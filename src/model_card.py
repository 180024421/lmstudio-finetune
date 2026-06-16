"""增强 Model Card 生成（含评测与 metrics）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from .config_loader import ROOT


def build_model_card_md(cfg: dict[str, Any], *, merged_dir: Path, adapter_path: Path | None = None) -> str:
    lcfg = cfg.get("lora") or {}
    tcfg = cfg.get("train") or {}
    eval_score = None
    eval_path = ROOT / "output" / "eval_report.json"
    if eval_path.exists():
        try:
            eval_score = json.loads(eval_path.read_text(encoding="utf-8")).get("avg_score")
        except Exception:
            pass

    metrics_tail = []
    run_dir = ROOT / cfg.get("output_dir", "output/run1")
    metrics_file = run_dir / "metrics.jsonl"
    if metrics_file.exists():
        for line in metrics_file.read_text(encoding="utf-8").strip().splitlines()[-5:]:
            try:
                metrics_tail.append(json.loads(line))
            except Exception:
                pass

    lines = [
        "# 微调模型 Model Card",
        "",
        f"- **基座**: {cfg.get('base_model')}",
        f"- **LoRA**: `{adapter_path or 'N/A'}`",
        f"- **合并时间**: {datetime.now(timezone.utc).isoformat()}",
        f"- **描述**: {lcfg.get('description', '')}",
        f"- **标签**: {', '.join(lcfg.get('tags') or [])}",
        "",
        "## 训练摘要",
        "",
        f"- epochs: {tcfg.get('num_epochs')}",
        f"- batch: {tcfg.get('per_device_train_batch_size')}",
        f"- lr: {tcfg.get('learning_rate')}",
        "",
    ]
    if eval_score is not None:
        lines.extend([f"## 评测", "", f"- 规则评测均分: **{eval_score:.3f}**", ""])
    if metrics_tail:
        lines.extend(["## 最近训练指标", "", "```json", json.dumps(metrics_tail, ensure_ascii=False, indent=2), "```", ""])
    lines.extend(
        [
            "## 部署",
            "",
            "- LM Studio: 加载 `output/model-q4_k_m.gguf`",
            "- Ollama: `ollama create my-model -f Modelfile`",
        ]
    )
    return "\n".join(lines)


def write_enhanced_model_card(merged_dir: Path, cfg: dict[str, Any], *, adapter_path: Path | None = None) -> Path:
    content = build_model_card_md(cfg, merged_dir=merged_dir, adapter_path=adapter_path)
    path = merged_dir / "MODEL_CARD.md"
    path.write_text(content, encoding="utf-8")
    return path
