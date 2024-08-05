# conf.py

# Import necessary modules
import os
import sys
from pathlib import Path
from sphinx_gallery.sorting import FileNameSortKey

# Path setup
sys.path.insert(0, str(Path(__file__).parents[1]))

import splineops

# Project information
project = 'splineops'
copyright = '2024, BIG lab'
author = 'BIG lab'
release = '0.0.1'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery',
    'sphinx-prompt',
    'sphinx_copybutton',
    'sphinx_remove_toctrees',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Options for HTML output
html_theme = 'pydata_sphinx_theme'

# Set html_static_path to an absolute path
html_static_path = [os.path.abspath('_static')]
html_css_files = ['css/custom.css']

# sphinx-gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': [os.path.abspath('../examples')],
    'gallery_dirs': [os.path.abspath('auto_examples')],
    'within_subsection_order': FileNameSortKey,
    'backreferences_dir': os.path.abspath('gen_modules/backreferences'),
    'filename_pattern': '.*',
    'matplotlib_animations': True,
}

html_theme_options = {
    'navbar_start': ['navbar-logo'],
    'navbar_center': ['navbar-nav'],
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'icon_links': [
        {
            'name': '',
            'url': 'https://github.com/Biomedical-Imaging-Group/splineops',
            'icon': 'fa-brands fa-github',
            'attributes': {'title': 'GitHub'},
        },
        {
            'name': '',
            'url': 'https://pypi.org/project/splineops/',
            'icon': 'fa-brands fa-python',
            'attributes': {'title': 'PyPI'},
        },
    ],
    'use_edit_page_button': True,
}

html_context = {
    'github_user': 'Biomedical-Imaging-Group',
    'github_repo': 'splineops',
    'github_version': 'main',
    'doc_path': 'doc',
}

# Ensure the paths to logo and favicon are absolute
html_logo = os.path.abspath('./_static/logo.png')
html_favicon = os.path.abspath('./_static/logo.ico')

# Function to ensure the static directory path is absolute
def resolve_static_path(app, exception):
    """
    Ensure the '_static' directory path is absolute.
    This function is necessary to avoid the assertion error in the
    'pydata_sphinx_theme.logo' extension, which requires 'staticdir'
    to be an absolute path.
    """
    staticdir = Path(app.builder.outdir) / "_static"
    if not staticdir.is_absolute():
        staticdir = staticdir.resolve()
    app.builder.outdir = str(staticdir.parent)  # Ensure outdir points to the correct directory

# Function to replace unpicklable objects with their qualified names
def make_sphinx_gallery_conf_picklable(app, config):
    """
    Replace unpicklable objects in 'sphinx_gallery_conf' with their
    fully qualified names to prevent warnings about unpicklable values.
    This ensures the configuration can be cached without issues.
    """
    new_conf = config.sphinx_gallery_conf.copy()
    # Replace 'within_subsection_order' with its fully qualified name if it's callable
    if 'within_subsection_order' in new_conf and callable(new_conf['within_subsection_order']):
        new_conf['within_subsection_order'] = f"{new_conf['within_subsection_order'].__module__}.{new_conf['within_subsection_order'].__name__}"
    config.sphinx_gallery_conf = new_conf

# Setup function called by Sphinx to connect event handlers
def setup(app):
    # Connect the 'resolve_static_path' function to the 'build-finished' event
    app.connect('build-finished', resolve_static_path)
    
    # Connect the 'make_sphinx_gallery_conf_picklable' function to the 'config-inited' event
    app.connect('config-inited', make_sphinx_gallery_conf_picklable)
    
    # Existing build-finished event setup for debugging
    def on_build_finished(app, exception):
        if exception:
            print(f"Build finished with exception: {exception}")
            import traceback
            traceback.print_exc()
        else:
            print("Build finished successfully")
    app.connect('build-finished', on_build_finished)
