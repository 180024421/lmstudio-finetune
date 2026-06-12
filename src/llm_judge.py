from __future__ import annotations

import json
import re
from typing import Any

from .config_loader import load_config
from .lmstudio_client import completion, openai_client

JUDGE_PROMPT = """你是严格的回答质量评审员。根据「问题」「参考答案」「模型回答」打分。

评分标准（1-5 整数）：
5 = 准确完整，可直接替代参考答案
4 = 基本正确，少量遗漏
3 = 部分相关，有明显缺漏或偏题
2 = 大部分错误或空洞
1 = 完全无关或胡编

只输出 JSON：{{"score": 4, "reason": "一句话说明"}}

问题：{prompt}
参考答案：{expected}
模型回答：{actual}
"""


def _parse_judge_json(text: str) -> tuple[int, str]:
    raw = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return 3, "无法解析评审 JSON"
    try:
        data = json.loads(m.group())
        score = int(data.get("score", 3))
        score = max(1, min(5, score))
        return score, str(data.get("reason", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 3, "评审 JSON 无效"


def judge_response(
    prompt: str,
    expected: str,
    actual: str,
    cfg: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """返回 0~1 归一化分数与理由。"""
    cfg = cfg or load_config()
    jcfg = cfg.get("eval") or {}
    lcfg = cfg.get("lm_studio") or {}
    _, default_model = openai_client(cfg)
    model = jcfg.get("judge_model") or lcfg.get("model") or default_model
    user = JUDGE_PROMPT.format(prompt=prompt[:1500], expected=expected[:1500], actual=actual[:1500])
    resp = completion(
        [{"role": "user", "content": user}],
        cfg,
        model=model,
        temperature=0.1,
        max_tokens=256,
        operation="judge",
    )
    content = resp.choices[0].message.content or ""
    score, reason = _parse_judge_json(content)
    return score / 5.0, reason
