# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile prepare live html clean

# 1. The Automation Pipeline: Replicates your GitHub Actions steps exactly
prepare:
	@echo "Preparing documentation workspace assets..."
	@mkdir -p source/logo
	@if [ -f ../logo/logo.png ]; then cp ../logo/logo.png source/logo/logo.png; fi
	@rm -f source/otaf*.rst source/modules.rst
	@echo "Converting and patching project README..."
	@pandoc ../README.md -f markdown -t rst -o source/README.rst
	@python3 -c '\
content = open("source/README.rst").read(); \
marker = "About The Project\n-----------------"; \
body = content.split(marker)[1] if marker in content else content; \
mangled_link = ".. raw:: html\n\n   <p align=\"right\">\n\n(back to top)\n\n.. raw:: html\n\n   </p>"; \
working_link = ".. raw:: html\n\n   <p align=\"right\"><a href=\"#readme-top\">back to top</a></p>"; \
body = body.replace(mangled_link, working_link); \
header = """.. _readme-top:\n\n|Contributors| |Forks| |Stargazers| |Issues| |MIT License|\n\n.. image:: logo/logo.png\n   :width: 240px\n   :align: center\n   :alt: OTAF Logo\n\n.. role:: center\n   :class: center\n\n**OTAF: Open Tolerance Analysis Framework**\n\nA scientific Python framework for statistical tolerance analysis of over-constrained mechanical assemblies using linear system-of-constraints optimization.\n\n`Explore the Docs <https://kramer84.github.io/otaf/>`_ | `Report Bug <https://github.com/Kramer84/otaf/issues>`_ | `Request Feature <https://github.com/Kramer84/otaf/issues>`_\n\n.. contents:: Table of Contents\n   :depth: 3\n\n----"""; \
open("source/README.rst", "w").write(header + "\n\n" + marker + body); \
'
	@echo "Running sphinx-apidoc discovery loop..."
	@sphinx-apidoc -f -o source/ ../src/otaf
	@echo "Writing automated subpage structural indices..."
	@python3 -c '\
open("source/otaf.example_models.models_2_D.rst", "w").write("otaf.example_models.models_2_D package\n======================================\n\n.. automodule:: otaf.example_models.models_2_D\n   :members:\n   :undoc-members:\n   :show-inheritance:\n\nAvailable Models\n----------------\n\n.. toctree::\n   :maxdepth: 1\n\n   otaf.example_models.models_2_D.model1\n   otaf.example_models.models_2_D.model2\n"); \
open("source/otaf.example_models.models_2_D.model1.rst", "w").write("Model 1: 2D 4-DOF Analytical Reference\n======================================\n\n.. automodule:: otaf.example_models.models_2_D.model1\n   :members:\n   :undoc-members:\n   :show-inheritance:\n"); \
open("source/otaf.example_models.models_2_D.model2.rst", "w").write("Model 2: 2D 16-DOF Reference Assembly\n=====================================\n\n.. automodule:: otaf.example_models.models_2_D.model2\n   :members:\n   :undoc-members:\n   :show-inheritance:\n"); \
open("source/otaf.example_models.models_3_D.rst", "w").write("otaf.example_models.models_3_D package\n======================================\n\n.. automodule:: otaf.example_models.models_3_D\n   :members:\n   :undoc-members:\n   :show-inheritance:\n\nAvailable Models\n----------------\n\n.. toctree::\n   :maxdepth: 1\n\n   otaf.example_models.models_3_D.model3\n   otaf.example_models.models_3_D.model4\n"); \
open("source/otaf.example_models.models_3_D.model3.rst", "w").write("Model 3: 3D 30-DOF Mechanical System\n====================================\n\n.. automodule:: otaf.example_models.models_3_D.model3\n   :members:\n   :undoc-members:\n   :show-inheritance:\n"); \
open("source/otaf.example_models.models_3_D.model4.rst", "w").write("Model 4: 50-DOF Complex Hyperplane Assembly\n============================================\n\n.. automodule:: otaf.example_models.models_3_D.model4\n   :members:\n   :undoc-members:\n   :show-inheritance:\n"); \
'

# 2. Intercepts standard manual builds to execute the pipeline targets first
html: prepare
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# 3. Your Local Live-Reload Shortcut: Runs preparation once, then spins up the watcher
live: prepare
	@sphinx-autobuild . _build/html --ignore "_build/*" --ignore "**/_build/*" --ignore "**.ipynb_checkpoints/*"

# Catch-all target for standard sphinx options
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)