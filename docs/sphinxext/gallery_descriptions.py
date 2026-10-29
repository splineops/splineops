"""Expose Sphinx-Gallery's example summaries as visible card descriptions."""

from pathlib import Path

from bs4 import BeautifulSoup


def add_gallery_descriptions(app, exception):
    """Reuse generated summaries rather than maintain a second caption list."""
    if exception is not None or app.builder.format != "html":
        return

    for gallery_dir in app.config.sphinx_gallery_conf["gallery_dirs"]:
        for page in (Path(app.builder.outdir) / gallery_dir).rglob("index.html"):
            soup = BeautifulSoup(page.read_text(encoding="utf-8"), "html.parser")
            changed = False
            for card in soup.select(".sphx-glr-thumbcontainer[tooltip]"):
                summary = card.get("tooltip", "").strip()
                if not summary:
                    continue
                description = card.select_one(".sphx-glr-thumbnail-description")
                if description is None:
                    description = soup.new_tag(
                        "p", attrs={"class": "sphx-glr-thumbnail-description"}
                    )
                    card.append(description)
                # Assigning text escapes markup and preserves accessible prose.
                description.string = summary
                del card["tooltip"]
                changed = True
            if changed:
                page.write_text(str(soup), encoding="utf-8")


def setup(app):
    # Run after Sphinx-Gallery's own HTML post-processing. Removing the tooltip
    # makes repeated incremental builds idempotent and avoids duplicate captions.
    app.connect("build-finished", add_gallery_descriptions, priority=910)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
