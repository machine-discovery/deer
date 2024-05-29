# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import sphinx_rtd_theme

FDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(FDIR, '../../deer/')))

# -- API docs file generators ------------------------------------------------
import json
import importlib
import inspect


# generate the api files
api_dir = os.path.abspath(os.path.join(FDIR, "api"))
api_toc_path = os.path.join(api_dir, "toc.json")
module_index_template = """{name}
{underlines}

.. toctree::
   :maxdepth: 1

   {api_list}
"""
module_api_list_indent = " " * 3
file_template = """{name}
{underlines}

.. auto{function_or_class}:: {fullname}
   {with_members}
"""
with open(api_toc_path, "r") as f:
    api_toc = json.load(f)

def format_api_display(api, desc):
    if desc == "":
        return api
    else:
        return "{api}: {desc} <{api}>".format(desc=desc, api=api)

for module, module_details in api_toc.items():
    api_list = [format_api_display(api, desc) for (api, desc) in module_details["api"].items()]
    module_dir_name = module.replace(".", "_")
    module_api_dir = os.path.join(api_dir, module_dir_name)
    if not os.path.exists(module_api_dir):
        os.mkdir(module_api_dir)

    # write the index.rst
    api_list_str = ("\n" + module_api_list_indent).join(api_list)
    module_index_fname = os.path.join(module_api_dir, "index.rst")
    with open(module_index_fname, "w") as f:
        content = module_index_template.format(
            name=module,
            underlines="=" * len(module),
            api_list=api_list_str
        )
        f.write(content)

    pymod = importlib.import_module(module)
    for fn in module_details["api"]:
        fullname = module + "." + fn
        isfn = inspect.isfunction(getattr(pymod, fn))
        fname = os.path.join(module_api_dir, fn + ".rst")
        file_content = file_template.format(
            name=fn,
            fullname=fullname,
            underlines="=" * len(fn),
            function_or_class="function" if isfn else "class",
            with_members="" if isfn else ":members:"
        )
        with open(fname, "w") as f:
            f.write(file_content)

# -- General configuration ------------------------------------------------

project = 'deer'
copyright = '2024, Machine Discovery Ltd'
author = 'Machine Discovery Ltd'

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

anonymize = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    # 'jupyter_sphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode'
]

napoleon_include_special_with_doc = True
napoleon_include_private_with_doc = True
autodoc_member_order = "bysource"

html_static_path = ['_static']
html_css_files = [
    'css/math.css',
    'css/general.css',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The master toctree document.
master_doc = 'index'

# # -- General configuration ---------------------------------------------------
# # https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

# templates_path = ['_templates']
# exclude_patterns = []



# # -- Options for HTML output -------------------------------------------------
# # https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = not anonymize
