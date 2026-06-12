from __future__ import annotations

import json
from pathlib import Path

from src.promo_integration import bridge_path, import_promo_with_job_split, write_bridge


def test_promo_job_split(tmp_path: Path):
    jobs = tmp_path / "jobs"
    for i in range(3):
        job = jobs / f"job{i}"
        job.mkdir(parents=True)
        (job / "transcript.txt").write_text(f"topic {i}", encoding="utf-8")
        (job / "promo_copy.json").write_text(
            json.dumps({"bilibili": {"titles": [f"title {i}"], "description": f"desc {i}"}}),
            encoding="utf-8",
        )
    out = tmp_path / "data"
    stats = import_promo_with_job_split(jobs, out, eval_job_ratio=0.34, seed=1)
    assert stats["jobs_total"] == 3
    assert stats["train_samples"] > 0
    assert stats["eval_samples"] > 0
    assert Path(stats["train_file"]).exists()
    assert Path(stats["eval_file"]).exists()


def test_write_bridge(tmp_path: Path):
    promo = tmp_path / "promo"
    promo.mkdir()
    path = write_bridge(promo, {"recommended_lora": "v1", "eval_score": 0.9})
    assert path == bridge_path(promo)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["recommended_lora"] == "v1"
