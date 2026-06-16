"""Webhook 通知（钉钉/飞书/企业微信/通用）。"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from .notify import notify


def send_webhook(url: str, title: str, message: str, *, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    if not url or not url.strip():
        notify(title, message)
        return {"ok": True, "channel": "desktop", "detail": "无 webhook URL，已回退桌面通知"}

    payload: dict[str, Any]
    if "dingtalk" in url or "oapi.dingtalk.com" in url:
        payload = {"msgtype": "text", "text": {"content": f"{title}\n{message}"}}
    elif "feishu" in url or "lark" in url:
        payload = {"msg_type": "text", "content": {"text": f"{title}\n{message}"}}
    else:
        payload = {"title": title, "message": message, "text": f"{title}\n{message}"}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return {"ok": True, "channel": "webhook", "status": resp.status, "body": body[:500]}
    except Exception as e:
        notify(title, message)
        return {"ok": False, "channel": "webhook", "error": str(e), "fallback": "desktop"}


def notify_training_complete(cfg: dict[str, Any], *, success: bool, detail: str = "") -> dict[str, Any]:
    wh = (cfg.get("monitoring") or {}).get("webhook_url", "")
    title = "训练完成" if success else "训练失败"
    msg = detail or ("QLoRA 训练已结束" if success else "请查看日志")
    return send_webhook(wh, title, msg, cfg=cfg)
