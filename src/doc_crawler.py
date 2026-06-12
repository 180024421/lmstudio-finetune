from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

SKIP_EXT = (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".css", ".js", ".svg", ".ico")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = True
        if tag in ("p", "h1", "h2", "h3", "h4", "li", "td", "pre", "blockquote"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self._chunks.append(text + " ")

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        return raw.strip()


def _fetch_html(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "lmstudio-finetune-crawler/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="ignore")


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def _normalize_url(base: str, link: str) -> str | None:
    if not link or link.startswith("#") or link.startswith("mailto:") or link.startswith("javascript:"):
        return None
    full = urllib.parse.urljoin(base, link)
    parsed = urllib.parse.urlparse(full)
    if parsed.scheme not in ("http", "https"):
        return None
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in SKIP_EXT):
        return None
    # strip fragment
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def extract_links(base_url: str, html: str) -> list[str]:
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out: list[str] = []
    seen: set[str] = set()
    for link in links:
        norm = _normalize_url(base_url, link)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def crawl_urls(
    start_urls: list[str],
    *,
    max_pages: int = 10,
    same_domain_only: bool = True,
    delay_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    queue = list(start_urls)
    visited: set[str] = set()
    pages: list[dict[str, Any]] = []
    base_domains = {urllib.parse.urlparse(u).netloc for u in start_urls}

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            html = _fetch_html(url)
        except Exception as e:
            console.print(f"[yellow]跳过 {url}: {e}[/yellow]")
            continue
        text = html_to_text(html)
        if len(text) < 80:
            continue
        title_m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title = title_m.group(1).strip() if title_m else url
        pages.append({"url": url, "title": title, "text": text})
        for link in extract_links(url, html):
            if link in visited:
                continue
            if same_domain_only and urllib.parse.urlparse(link).netloc not in base_domains:
                continue
            queue.append(link)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return pages


def crawl_to_markdown_files(
    start_urls: list[str],
    output_dir: Path,
    *,
    max_pages: int = 10,
    same_domain_only: bool = True,
    delay_seconds: float = 1.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = crawl_urls(
        start_urls,
        max_pages=max_pages,
        same_domain_only=same_domain_only,
        delay_seconds=delay_seconds,
    )
    paths: list[str] = []
    for i, page in enumerate(pages):
        safe = re.sub(r"[^\w\-]+", "_", page["title"])[:40] or f"page_{i}"
        path = output_dir / f"{i:03d}_{safe}.md"
        body = f"# {page['title']}\n\n来源: {page['url']}\n\n{page['text']}\n"
        path.write_text(body, encoding="utf-8")
        paths.append(str(path))
    return {"pages": len(pages), "output_dir": str(output_dir), "files": paths}
