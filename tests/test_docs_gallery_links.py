"""Check local gallery downloads with and without the optional Binder badge.

These integration checks use the documentation extras, installed by the quality
workflow. Numerical-only test environments can skip them.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sphinx")
BeautifulSoup = pytest.importorskip("bs4").BeautifulSoup


@pytest.mark.parametrize("with_binder", [False, True])
def test_gallery_downloads_do_not_require_binder(tmp_path, with_binder):
    extension_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "sphinxext"
        / "move_gallery_links.py"
    )
    spec = importlib.util.spec_from_file_location("gallery_links_test", extension_path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)

    gallery = tmp_path / "auto_examples"
    gallery.mkdir()
    page = gallery / "example.html"
    badge = (
        '<div class="binder-badge"><a href="https://mybinder.org/example">'
        '<img src="binder.svg" alt="Launch Binder"></a></div>'
        if with_binder
        else ""
    )
    page.write_text(
        '<html><body><div class="sidebar-secondary-items"></div>'
        '<div class="sphx-glr-download-link-note">See the downloads below.</div>'
        '<div class="sphx-glr-footer">'
        '<div class="sphx-glr-download-python">'
        '<a href="example.py"><span>example.py</span></a></div>'
        '<div class="sphx-glr-download-jupyter">'
        '<a href="example.ipynb"><span>example.ipynb</span></a></div>'
        f"{badge}</div></body></html>",
        encoding="utf-8",
    )
    app = SimpleNamespace(
        config=SimpleNamespace(sphinx_gallery_conf={"gallery_dirs": ["auto_examples"]}),
        builder=SimpleNamespace(outdir=str(tmp_path)),
        project=SimpleNamespace(docnames={"auto_examples/example"}),
        verbosity=0,
    )

    extension.move_gallery_links(app, exception=None)

    rendered = BeautifulSoup(page.read_text(encoding="utf-8"), "html.parser")
    sidebar = rendered.select_one(".sidebar-secondary-items")
    downloads = {link["href"] for link in sidebar.select("a[download]")}
    assert downloads == {"example.py", "example.ipynb"}
    assert sidebar.get_text(" ", strip=True).startswith("Download source code")
    binder_links = sidebar.select('a[href="https://mybinder.org/example"]')
    assert len(binder_links) == int(with_binder)
    assert rendered.select_one(".sphx-glr-footer") is None
    assert rendered.select_one(".sphx-glr-download-link-note") is None
