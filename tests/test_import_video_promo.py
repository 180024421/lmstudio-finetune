from __future__ import annotations

import json
from pathlib import Path

from src.import_video_promo import import_job_dir


def test_import_promo_copy(tmp_path: Path):
    job = tmp_path / "job1"
    job.mkdir()
    (job / "transcript.txt").write_text("Redis 缓存穿透讲解", encoding="utf-8")
    (job / "promo_copy.json").write_text(json.dumps({
        "bilibili": {"titles": ["Redis 穿透怎么防"], "description": "简介"},
    }), encoding="utf-8")
    rows = import_job_dir(job)
    assert len(rows) >= 2
    assert any("B 站" in r.get("instruction", "") for r in rows)
