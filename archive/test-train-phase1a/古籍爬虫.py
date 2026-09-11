#!/usr/bin/env python3
"""Scrape shuge.org for ancient Chinese texts into data/ancient_texts/"""

import os
import sys
import time
import random
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

RETRY = 3
TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = (4, 8)

BASE_URL = "https://www.shuge.org/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

OUTPUT_DIR = Path("/home/dja/桌面/Test Train OSC-LLM/data/ancient_texts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update(HEADERS)


def safe_get(url):
    for attempt in range(RETRY):
        try:
            time.sleep(random.uniform(*SLEEP_BETWEEN_REQUESTS))
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code == 503 and attempt < RETRY - 1:
                continue
            resp.raise_for_status()
            return resp
        except (requests.RequestException,) as e:
            print(f"  Request failed ({attempt+1}/{RETRY}): {e}")
            if attempt == RETRY - 1:
                raise
            time.sleep(2)
    return None


def extract_urls(html, selector="a"):
    from html.parser import HTMLParser

    urls = []

    class LinkParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for attr, value in attrs:
                    if attr == "href":
                        full_url = urljoin(BASE_URL, value)
                        urls.append(full_url)

    parser = LinkParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    return list(set(urls))


def is_content_page(url):
    paths = ["/html/", "/ang dai/", "/ang shi/"]
    return any(p in url.lower() for p in paths)


def extract_text_from_html(html):
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text_parts = []
            self.in_content = False
            self._current = []

        def handle_starttag(self, tag, attrs):
            if tag == "div":
                for a, v in attrs:
                    if a == "id" and "content" in v.lower():
                        self.in_content = True
                    elif a == "id" and "read" in v.lower():
                        self.in_content = True

        def handle_endtag(self, tag):
            if tag == "div":
                self.in_content = False

        def handle_data(self, data):
            if self.in_content:
                cleaned = data.strip()
                if cleaned:
                    self.text_parts.append(cleaned)

    extractor = TextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        pass
    return "\n".join(extractor.text_parts)


def scrape_shuge():
    print(f"[1/4] Fetching homepage: {BASE_URL}")
    try:
        resp = safe_get(BASE_URL)
        home_html = resp.text if resp else ""
    except Exception as e:
        print(f"Failed to fetch homepage: {e}")
        return

    category_urls = []
    all_links = extract_urls(home_html)

    for url in all_links[:100]:
        if "/ang/" in url.lower() or "/ang dai" in url.lower():
            category_urls.append(url)

    category_urls = list(set(category_urls))[:6]
    print(f"[2/4] Found {len(category_urls)} categories")

    processed_files = set()

    for cat_url in category_urls:
        print(f"[3/4] Processing category: {cat_url}")
        try:
            resp = safe_get(cat_url)
            if not resp:
                continue
            cat_html = resp.text

            sub_links = extract_urls(cat_html)
            detail_urls = [u for u in sub_links if is_content_page(u)][:20]

            for detail_url in detail_urls:
                if detail_url in processed_files:
                    continue
                try:
                    print(f"    Downloading: {detail_url}")
                    resp = safe_get(detail_url)
                    if not resp:
                        continue

                    content_html = resp.text
                    text = extract_text_from_html(content_html)

                    if not text or len(text) < 20:
                        continue

                    filename = (
                        detail_url.split("/")[-1]
                        .replace(".html", "")
                        .replace(" ", "_")
                        + ".txt"
                    )
                    out_path = OUTPUT_DIR / filename
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(text)

                    processed_files.add(detail_url)
                    print(f"      Saved: {filename} ({len(text)} chars)")
                except Exception as e:
                    print(f"    Error on detail page: {e}")
        except Exception as e:
            print(f"  Error on category: {e}")

    if not processed_files:
        fallback_text = """【示例文本】
天地玄黄，宇宙洪荒。
日月盈昃，辰宿列张。
寒来暑往，秋收冬藏。
闰余成岁，律吕调阳。

——《千字文》
"""
        out_path = OUTPUT_DIR / "fallback_sample.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(fallback_text)
        print("[WARN] No pages scraped; wrote fallback sample")

    total_txt = sum(1 for _ in OUTPUT_DIR.glob("*.txt"))
    print(f"[4/4] Done. {total_txt} files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    scrape_shuge()
