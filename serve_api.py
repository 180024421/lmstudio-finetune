#!/usr/bin/env python3
"""启动 FastAPI 推理服务（proxy 转发 LM Studio 或 local 加载 HF+LoRA）。"""

from __future__ import annotations

from src.api_server import main

if __name__ == "__main__":
    main()
