from __future__ import annotations

from src.doc_crawler import extract_links, html_to_text


def test_html_to_text():
    html = "<html><body><h1>Title</h1><p>Hello world</p><script>ignore</script></body></html>"
    text = html_to_text(html)
    assert "Hello world" in text
    assert "ignore" not in text


def test_extract_links():
    html = '<a href="/page2">x</a><a href="https://other.com/x">y</a>'
    links = extract_links("https://example.com/page1", html)
    assert "https://example.com/page2" in links
