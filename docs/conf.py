# splineops/docs/conf.py

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

from sphinx_gallery.sorting import FileNameSortKey

# -----------------------------------------------------------------------------
# Prefer native if available, but don't break builds if it isn't
# -----------------------------------------------------------------------------
import importlib.util

has_native = importlib.util.find_spec("splineops._lsresize") is not None
if "SPLINEOPS_ACCEL" not in os.environ:  # allow users/CI to override
    os.environ["SPLINEOPS_ACCEL"] = "always" if has_native else "auto"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)

DOCS_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Build-only generated static directory (NOT tracked by git)
# Sphinx will copy it into: docs/_build/html/_static/
# -----------------------------------------------------------------------------
GENERATED_STATIC_DIR = DOCS_DIR / "_build" / "_generated_static"
GENERATED_STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Optional: clean only generated animations each build start (avoids stale files)
anim_gen_dir = GENERATED_STATIC_DIR / "animations"
if anim_gen_dir.exists():
    shutil.rmtree(anim_gen_dir)
anim_gen_dir.mkdir(parents=True, exist_ok=True)

# IMPORTANT: set these EARLY so sphinx-gallery examples can export during execution
os.environ["SPLINEOPS_SPHINX_BUILD"] = "1"
os.environ["SPLINEOPS_SPHINX_STATICDIR"] = str(GENERATED_STATIC_DIR)

# -----------------------------------------------------------------------------
# FFMPEG setup for Matplotlib animations (Windows-friendly)
# -----------------------------------------------------------------------------
import matplotlib as mpl
from matplotlib import animation as mpl_animation

SG_ANIM_FORMAT = "jshtml"  # fallback if ffmpeg writer isn't available

try:
    import imageio_ffmpeg

    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    print("[docs] imageio-ffmpeg exe:", ffmpeg_exe)

    if ffmpeg_exe.exists():
        # Smoke test
        subprocess.run(
            [str(ffmpeg_exe), "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Create PATH-visible alias "ffmpeg(.exe)" under docs/_build/_ffmpeg_bin
        ffmpeg_alias_dir = DOCS_DIR / "_build" / "_ffmpeg_bin"
        ffmpeg_alias_dir.mkdir(parents=True, exist_ok=True)

        alias_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        ffmpeg_alias = ffmpeg_alias_dir / alias_name
        if not ffmpeg_alias.exists():
            shutil.copyfile(ffmpeg_exe, ffmpeg_alias)

        os.environ["PATH"] = str(ffmpeg_alias_dir) + os.pathsep + os.environ.get("PATH", "")

        # Force Matplotlib to use the alias name
        mpl.rcParams["animation.ffmpeg_path"] = "ffmpeg"
        mpl.rcParams["animation.writer"] = "ffmpeg"
        mpl.rcParamsDefault["animation.ffmpeg_path"] = "ffmpeg"
        mpl.rcParamsDefault["animation.writer"] = "ffmpeg"

        print("[docs] writers.is_available('ffmpeg'):", mpl_animation.writers.is_available("ffmpeg"))
        if mpl_animation.writers.is_available("ffmpeg"):
            SG_ANIM_FORMAT = "html5"
except Exception as e:
    print("[docs] ffmpeg setup failed; using jshtml. Reason:", e)

# -----------------------------------------------------------------------------
# Import the installed/editable package
# -----------------------------------------------------------------------------
import splineops  # noqa: E402

# Keep custom extensions path
sys.path.insert(0, os.path.abspath("sphinxext"))

def _log_native():
    import importlib.util as _util
    try:
        import splineops as _s
        print("[docs] splineops from:", getattr(_s, "__file__", "<unknown>"))
        print("[docs] native present:", _util.find_spec("splineops._lsresize") is not None)
        print("[docs] OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
        print("[docs] export static dir:", os.environ.get("SPLINEOPS_SPHINX_STATICDIR"))
    except Exception as e:
        print("[docs] import failure:", e)

_log_native()

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "splineops"
copyright = f"{datetime.now().year}, SplineOps authors"
release = splineops.__version__

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
    "sphinx-prompt",
    "sphinx_copybutton",
    "sphinx_remove_toctrees",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "myst_parser",
    "move_gallery_links",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "**sg_execution_times.rst",
]

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"{project} Documentation"

# IMPORTANT: include GENERATED_STATIC_DIR so its contents get copied to final _static/
html_static_path = ["_static", str(GENERATED_STATIC_DIR)]
html_css_files = ["css/custom.css"]

sg_examples_dir = "../examples"
sg_gallery_dir = "auto_examples"

sphinx_gallery_conf = {
    "examples_dirs": [sg_examples_dir],
    "gallery_dirs": [sg_gallery_dir],
    "within_subsection_order": FileNameSortKey,
    "backreferences_dir": "gen_modules/backreferences",
    "filename_pattern": ".*",
    "matplotlib_animations": (True, SG_ANIM_FORMAT),
    "binder": {
        "org": "splineops",
        "repo": "splineops.github.io",
        "binderhub_url": "https://mybinder.org",
        "branch": "main",
        "dependencies": "../.binder/requirements.txt",
        "notebooks_dir": "notebooks_binder",
        "use_jupyter_lab": True,
    },
    "remove_config_comments": True,
}

html_theme_options = {
    "logo": {"text": "SplineOps"},
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {"name": "", "url": "https://github.com/splineops/splineops", "icon": "fa-brands fa-github",
         "attributes": {"title": "GitHub"}},
        {"name": "", "url": "https://pypi.org/project/splineops/", "icon": "fa-brands fa-python",
         "attributes": {"title": "PyPI"}},
    ],
    "use_edit_page_button": True,
    "secondary_sidebar_items": {"**": ["page-toc", "sourcelink"]},
}

html_context = {
    "github_user": "splineops",
    "github_repo": "splineops",
    "github_version": "main",
    "doc_path": "docs",
}

html_logo = "_static/logo.png"
html_favicon = "_static/logo.ico"

# Hide secondary sidebar for sphinx-gallery index pages
html_theme_options["secondary_sidebar_items"][f"{sg_gallery_dir}/index"] = []
for sub_sg_dir in (Path(".") / sg_examples_dir).iterdir():
    if sub_sg_dir.is_dir():
        html_theme_options["secondary_sidebar_items"][f"{sg_gallery_dir}/{sub_sg_dir.name}/index"] = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "NumPy [stable]": ("https://numpy.org/doc/stable/", None),
    "CuPy [latest]": ("https://docs.cupy.dev/en/latest/", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/", None),
    "Pytest [latest]": ("https://docs.pytest.org/en/latest/", None),
    "Matplotlib [stable]": ("https://matplotlib.org/stable/", None),
}

def make_sphinx_gallery_conf_picklable(app, config):
    new_conf = config.sphinx_gallery_conf.copy()
    if "within_subsection_order" in new_conf and callable(new_conf["within_subsection_order"]):
        new_conf["within_subsection_order"] = (
            f"{new_conf['within_subsection_order'].__module__}."
            f"{new_conf['within_subsection_order'].__name__}"
        )
    config.sphinx_gallery_conf = new_conf

def setup(app):
    app.connect("config-inited", make_sphinx_gallery_conf_picklable)

# To build docs:
# pip install -e .[docs]
# cd docs
# make html
# Open `docs/_build/html/index.html`
