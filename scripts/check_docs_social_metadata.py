"""Check the rendered share cards, including nested pages and generated indexes.

Usage: python scripts/check_docs_social_metadata.py docs/_build/html
Uses Pillow, included in the documentation dependencies; run after an HTML build.
"""

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from PIL import Image


class HeadMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_head = False
        self.in_title = False
        self.meta = {}
        self.canonical = []
        self.title = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "head":
            self.in_head = True
        if not self.in_head:
            return
        if tag == "meta":
            key = attrs.get("property") or attrs.get("name")
            self.meta.setdefault(key, []).append(attrs.get("content", ""))
        elif tag == "link" and attrs.get("rel") == "canonical":
            self.canonical.append(attrs.get("href"))
        elif tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag == "head":
            self.in_head = False
        elif tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def check(build, base_url="https://splineops.github.io/"):
    count = 0
    images = {}
    for page in sorted(build.rglob("*.html")):
        relative = page.relative_to(build)
        # Media wrappers and theme assets are not documentation pages.
        if relative.parts[0].startswith("_"):
            continue
        head = HeadMetadata()
        head.feed(page.read_text(encoding="utf-8"))
        required = (
            "description",
            "og:title",
            "og:description",
            "og:type",
            "og:site_name",
            "og:url",
            "og:image",
            "og:image:secure_url",
            "og:image:type",
            "og:image:width",
            "og:image:height",
            "og:image:alt",
            "twitter:card",
            "twitter:title",
            "twitter:description",
            "twitter:image",
            "twitter:image:alt",
        )
        for key in required:
            values = head.meta.get(key, [])
            assert len(values) == 1 and values[0].strip(), (relative, key, values)
        meta = {k: v[0] for k, v in head.meta.items()}
        expected_url = urljoin(
            base_url, "" if relative.as_posix() == "index.html" else relative.as_posix()
        )
        assert head.canonical == [expected_url], (relative, head.canonical)
        assert meta["og:url"] == expected_url, relative
        assert meta["og:site_name"] == "SplineOps", relative
        assert meta["og:type"] == "website", relative
        assert meta["twitter:card"] == "summary_large_image", relative
        for field in ("title", "description", "image", "image:alt"):
            assert meta[f"twitter:{field}"] == meta[f"og:{field}"], (relative, field)
        assert meta["description"] == meta["og:description"], relative
        assert meta["og:image"] == meta["og:image:secure_url"], relative
        assert meta["og:image"].startswith(base_url), relative
        assert meta["og:image:type"] in ("image/png", "image/jpeg"), relative
        assert (meta["og:image:width"], meta["og:image:height"]) == ("1200", "630")
        assert head.title and " | SplineOps" not in head.title, relative
        assert "splineops 2.3.0 documentation" not in head.title, relative
        if relative.as_posix() != "index.html":
            assert meta["og:title"] == f"{head.title} | SplineOps", relative
        images[meta["og:image"]] = meta["og:image:type"]
        count += 1
    assert count, "No documentation pages found"
    for url, mime_type in images.items():
        image = build / urlsplit(url).path.lstrip("/")
        with Image.open(image) as decoded:
            assert decoded.size == (1200, 630), image
            assert Image.MIME[decoded.format] == mime_type, image
            decoded.verify()
        assert image.stat().st_size < 300_000, (image, image.stat().st_size)
    print(
        f"Verified share metadata on {count} pages and {len(images)} preview image(s)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    check(parser.parse_args().build)
