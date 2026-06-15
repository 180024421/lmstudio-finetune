from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from .chat_templates import row_to_text
from .data_dedup import deduplicate_rows
from .data_io import load_rows, write_jsonl

console = Console()

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except ImportError:
            _embedder = False
    return _embedder if _embedder is not False else None


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def semantic_deduplicate_rows(
    rows: list[dict[str, Any]],
    *,
    template: str = "qwen",
    threshold: float = 0.92,
    fallback_jaccard: float = 0.85,
) -> tuple[list[dict[str, Any]], int, str]:
    model = _get_embedder()
    if model is None:
        console.print("[yellow]未安装 sentence-transformers，回退字符去重[/yellow]")
        kept, removed = deduplicate_rows(rows, template=template, threshold=fallback_jaccard)
        return kept, removed, "jaccard"

    texts = [row_to_text(r, template)["text"] for r in rows]
    embs = model.encode(texts, show_progress_bar=False)
    kept: list[dict[str, Any]] = []
    kept_embs: list[list[float]] = []
    removed = 0
    for row, emb in zip(rows, embs):
        dup = False
        for prev in kept_embs:
            if _cosine(list(emb), list(prev)) >= threshold:
                dup = True
                break
        if dup:
            removed += 1
        else:
            kept.append(row)
            kept_embs.append(list(emb))
    return kept, removed, "embedding"


def semantic_deduplicate_file(
    input_path: Path,
    output_path: Path,
    *,
    template: str = "qwen",
    threshold: float = 0.92,
) -> dict[str, Any]:
    rows = load_rows(input_path)
    kept, removed, mode = semantic_deduplicate_rows(rows, template=template, threshold=threshold)
    write_jsonl(output_path, kept)
    return {"input": len(rows), "kept": len(kept), "removed": removed, "mode": mode}
