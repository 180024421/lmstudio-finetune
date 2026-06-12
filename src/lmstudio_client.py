from __future__ import annotations

import json
from typing import Any, Iterator

from openai import OpenAI
from rich.console import Console

from .api_usage import record_from_response
from .config_loader import load_config

console = Console()


def openai_client(cfg: dict[str, Any] | None = None) -> tuple[OpenAI, str]:
    cfg = cfg or load_config()
    lcfg = cfg.get("lm_studio") or {}
    client = OpenAI(
        base_url=lcfg.get("base_url", "http://127.0.0.1:1234/v1"),
        api_key=lcfg.get("api_key", "lm-studio"),
    )
    model = lcfg.get("model") or "local-model"
    return client, model


def _client(cfg: dict[str, Any]) -> tuple[OpenAI, str]:
    return openai_client(cfg)


def completion(
    messages: list[dict[str, str]],
    cfg: dict[str, Any] | None = None,
    *,
    model: str | None = None,
    operation: str = "chat",
    **kwargs: Any,
) -> Any:
    cfg = cfg or load_config()
    client, default_model = openai_client(cfg)
    use_model = model or default_model
    resp = client.chat.completions.create(model=use_model, messages=messages, **kwargs)
    record_from_response(resp, operation=operation, model=use_model, cfg=cfg)
    return resp


def chat(prompt: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    resp = completion(
        [{"role": "user", "content": prompt}],
        cfg,
        temperature=0.7,
        max_tokens=1024,
        operation="chat",
    )
    return resp.choices[0].message.content or ""


def chat_with_system(user: str, system: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    resp = completion(messages, cfg, temperature=0.7, max_tokens=1024, operation="chat")
    return resp.choices[0].message.content or ""


def chat_stream(prompt: str, cfg: dict[str, Any] | None = None) -> Iterator[str]:
    cfg = cfg or load_config()
    client, model = openai_client(cfg)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            payload = {"choices": [{"delta": {"content": delta}, "index": 0}]}
            yield json.dumps(payload, ensure_ascii=False)


def list_models(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_config()
    client, _ = openai_client(cfg)
    try:
        return [m.id for m in client.models.list().data]
    except Exception:
        return []


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
