"""Provide crawler-readable share metadata without changing browser tab titles."""

import re
from urllib.parse import urljoin

from docutils import nodes

SITE_DESCRIPTION = (
    "Turn sampled data into continuous spline models. Interpolate, resize with "
    "projection-based antialiasing, transform, and differentiate in N dimensions."
)
PAGE_DESCRIPTIONS = {
    "index": SITE_DESCRIPTION,
    "examples": (
        "Explore runnable examples of spline interpolation, antialiased "
        "resizing, affine transforms, smoothing, derivatives, and multiscale analysis."
    ),
    "modules": (
        "Find the right SplineOps tool for your scientific data: interpolation, "
        "resizing, geometry, derivatives, smoothing, regression, and multiscale analysis."
    ),
    "api/index": (
        "Explore the SplineOps API: interpolation, resizing, affine transforms, "
        "derivatives, smoothing, regression, and multiscale analysis."
    ),
}
GENERATED_TITLES = {
    "search": "Search",
    "genindex": "Index",
    "py-modindex": "Python Module Index",
}


def page_description(doctree):
    """Prefer an authored description, then the first substantial prose paragraph."""
    if doctree is not None:
        for meta in doctree.findall(nodes.meta):
            if meta.get("name") == "description" and meta.get("content"):
                return meta["content"]
        for paragraph in doctree.findall(nodes.paragraph):
            parent = paragraph.parent
            while parent is not None and not isinstance(
                parent, (nodes.Admonition, nodes.sidebar)
            ):
                parent = parent.parent
            text = " ".join(paragraph.astext().split())
            if parent is None and len(text) >= 60:
                # A complete introductory sentence reads better than a cut-off
                # paragraph in a compact messaging or social-media card.
                first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
                if 60 <= len(first_sentence) <= 200:
                    return first_sentence
                return text if len(text) <= 200 else text[:197].rsplit(" ", 1)[0] + "…"
    return SITE_DESCRIPTION


def add_social_metadata(app, pagename, templatename, context, doctree):
    base_url = app.config.html_baseurl.rstrip("/") + "/"
    title = context.get("title") or GENERATED_TITLES.get(pagename, pagename)
    # Make / and /index.html identify the same homepage when shared or indexed.
    context["pageurl"] = urljoin(
        base_url, "" if pagename == "index" else app.builder.get_target_uri(pagename)
    )
    context["social_title"] = (
        "SplineOps | From samples to continuous models"
        if pagename == "index"
        else f"{title} | SplineOps"
    )
    context["social_description"] = PAGE_DESCRIPTIONS.get(
        pagename, page_description(doctree)
    )
    context["social_image"] = urljoin(
        base_url, "_static/social/splineops-spline-art.jpg"
    )
    context["social_image_alt"] = (
        "SplineOps with Annette Unser's SIAM Cover Page artwork: spline curves "
        "and a fractional-spline formula on a blue watercolor background."
    )
    # A document's own .. meta:: description is already emitted by Sphinx.
    context["has_meta_description"] = doctree is not None and any(
        meta.get("name") == "description" for meta in doctree.findall(nodes.meta)
    )


def setup(app):
    app.connect("html-page-context", add_social_metadata)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
