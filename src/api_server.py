from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config_loader import load_config
from .lmstudio_client import chat, chat_stream, check_lm_studio, list_models

app = FastAPI(title="lmstudio-finetune API", version="1.0.0")

_model_cache: dict[str, Any] = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "local"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False


def _mode() -> str:
    return os.environ.get("API_MODE", "proxy")  # proxy | local


def _load_local_model(cfg: dict[str, Any]):
    if _model_cache.get("model"):
        return _model_cache["model"], _model_cache["tokenizer"]
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = cfg["base_model"]
    adapter = cfg.get("api", {}).get("adapter_dir", "")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if adapter:
        from pathlib import Path
        from .config_loader import ROOT

        p = Path(adapter)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            model = PeftModel.from_pretrained(model, str(p))
    _model_cache["model"] = model
    _model_cache["tokenizer"] = tokenizer
    return model, tokenizer


def _local_chat(messages: list[dict], cfg: dict, max_tokens: int) -> str:
    import torch

    model, tokenizer = _load_local_model(cfg)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


@app.get("/v1/models")
def models():
    cfg = load_config()
    if _mode() == "proxy":
        ids = list_models(cfg)
        return {"data": [{"id": i, "object": "model"} for i in ids] or [{"id": "local-model", "object": "model"}]}
    return {"data": [{"id": cfg.get("base_model", "local"), "object": "model"}]}


@app.get("/health")
def health():
    cfg = load_config()
    ok = check_lm_studio(cfg) if _mode() == "proxy" else True
    return {"status": "ok" if ok else "degraded", "mode": _mode()}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    cfg = load_config()
    messages = [m.model_dump() for m in req.messages]
    if not messages:
        raise HTTPException(400, "messages 不能为空")

    if req.stream:
        if _mode() != "proxy":
            raise HTTPException(400, "local 模式暂不支持 stream")
        user = messages[-1]["content"]

        def gen():
            for chunk in chat_stream(user, cfg):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    if _mode() == "proxy":
        if not check_lm_studio(cfg):
            raise HTTPException(503, "LM Studio 未连接")
        content = chat(messages[-1]["content"], cfg)
    else:
        content = _local_chat(messages, cfg, req.max_tokens)

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": content}, "index": 0}],
    }


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.api_server:app", host="127.0.0.1", port=port, reload=False)
