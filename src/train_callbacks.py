from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from .notify import notify


class MetricsJsonlCallback(TrainerCallback):
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {"step": state.global_step, **logs}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class BestAdapterCallback(TrainerCallback):
    def __init__(self, adapter_dir: Path, metric: str = "eval_loss") -> None:
        self.adapter_dir = adapter_dir
        self.best_dir = adapter_dir.parent / "best_adapter"
        self.metric = metric
        self.best: float | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric not in metrics:
            return
        value = float(metrics[self.metric])
        if self.best is None or value < self.best:
            self.best = value
            trainer = kwargs.get("model")
            if trainer is None:
                return
            # kwargs may contain model from trainer - use trainer from state
        # Save via trainer in on_save is tricky; use on_evaluate with model in kwargs
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        if model is None:
            return
        self.best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.best_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(self.best_dir)


class TrainCompleteNotifyCallback(TrainerCallback):
    def __init__(self, title: str = "lmstudio-finetune") -> None:
        self.title = title

    def on_train_end(self, args, state, control, **kwargs):
        notify(self.title, f"训练完成，共 {state.global_step} steps")


class CopyBestFromCheckpointCallback(TrainerCallback):
    """训练结束后将 load_best_model_at_end 的最佳权重额外复制到 best_adapter。"""

    def __init__(self, output_dir: Path, best_adapter_dir: Path) -> None:
        self.output_dir = output_dir
        self.best_adapter_dir = best_adapter_dir

    def on_train_end(self, args, state, control, **kwargs):
        if state.best_model_checkpoint:
            src = Path(state.best_model_checkpoint)
            if src.exists():
                if self.best_adapter_dir.exists():
                    shutil.rmtree(self.best_adapter_dir)
                shutil.copytree(src, self.best_adapter_dir)
