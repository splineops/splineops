# splineops/src/splineops/utils/sphinx.py
"""
splineops.utils.sphinx
======================

Small helpers meant only for Sphinx/Sphinx-Gallery builds.

Design goals
------------
- stdlib-only (no Matplotlib import here)
- no-op when not building docs
- write exported artifacts into the Sphinx *build* static dir
  (e.g. docs/_build/html/_static/...)

Environment contract (set by docs/conf.py)
------------------------------------------
- SPLINEOPS_SPHINX_BUILD=1
- SPLINEOPS_SPHINX_STATICDIR=<sphinx outdir>/_static
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

__all__ = ["get_sphinx_staticdir", "export_animation_mp4_and_html"]


def get_sphinx_staticdir() -> Optional[Path]:
    """Return the Sphinx build _static directory, or None if not in a doc build."""
    if os.environ.get("SPLINEOPS_SPHINX_BUILD") != "1":
        return None
    p = os.environ.get("SPLINEOPS_SPHINX_STATICDIR")
    return Path(p) if p else None


def export_animation_mp4_and_html(
    ani: Any,
    *,
    stem: str,
    interval_ms: float,
    dpi: int = 80,
    subdir: str = "animations",
    writer: str = "ffmpeg",
    autoplay: bool = True,
    loop: bool = True,
    muted: bool = True,
    controls: bool = True,
    playsinline: bool = True,
    force: bool = False,
) -> Optional[tuple[Path, Path]]:
    """
    Export a Matplotlib animation to MP4 + a responsive HTML wrapper.

    Writes to:
        <SPLINEOPS_SPHINX_STATICDIR>/<subdir>/<stem>.mp4
        <SPLINEOPS_SPHINX_STATICDIR>/<subdir>/<stem>.html

    Does nothing (returns None) unless building docs (env vars set by conf.py).
    """
    static_dir = get_sphinx_staticdir()
    if static_dir is None:
        return None

    out_dir = static_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / f"{stem}.mp4"
    html_path = out_dir / f"{stem}.html"

    # Keep timing consistent with the interactive animation interval
    fps = max(0.1, 1000.0 / float(interval_ms))  # float keeps 750ms -> 1.333fps

    try:
        if force or not mp4_path.exists():
            ani.save(str(mp4_path), writer=writer, fps=fps, dpi=dpi)
    except Exception as e:
        # Don’t kill doc builds if ffmpeg isn’t usable on some platform.
        # (Optionally write a tiny HTML placeholder.)
        html_path.write_text(
            "<!doctype html><html><body><p>"
            "Animation export failed during doc build.</p></body></html>",
            encoding="utf-8",
        )
        return None

    attrs = []
    if controls:
        attrs.append("controls")
    if loop:
        attrs.append("loop")
    if autoplay:
        attrs.append("autoplay")
    if muted:
        attrs.append("muted")
    if playsinline:
        attrs.append("playsinline")
    attrs_str = " ".join(attrs)

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body {{ margin: 0; padding: 0; }}
    video {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <video {attrs_str}>
    <source src="{mp4_path.name}" type="video/mp4">
  </video>
</body>
</html>
"""

    if force or not (html_path.exists() and html_path.read_text(encoding="utf-8") == html_doc):
        html_path.write_text(html_doc, encoding="utf-8")

    return mp4_path, html_path
