from __future__ import annotations

from typing import Any

from openai import OpenAI
from rich.console import Console

from .config_loader import load_config

console = Console()


def chat(prompt: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    lcfg = cfg.get("lm_studio") or {}
    client = OpenAI(
        base_url=lcfg.get("base_url", "http://127.0.0.1:1234/v1"),
        api_key=lcfg.get("api_key", "lm-studio"),
    )
    model = lcfg.get("model") or None
    resp = client.chat.completions.create(
        model=model or "local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


def check_lm_studio(cfg: dict[str, Any] | None = None) -> bool:
    cfg = cfg or load_config()
    lcfg = cfg.get("lm_studio") or {}
    import urllib.request

    url = lcfg.get("base_url", "http://127.0.0.1:1234/v1").rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False
