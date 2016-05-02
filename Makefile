VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PIP_COMPILER = $(VENV)/bin/pip-compile
PIP_INSTALL = $(PIP) install --exists-action=w

setup: venv deps

deps: venv
	@$(PIP_COMPILER) reqs.in -o reqs.txt
	@$(PIP_INSTALL) -r reqs.txt

venv:
	@virtualenv $(VENV) --prompt '<venv:connections>'
	@$(PIP_INSTALL) -U pip setuptools pip-tools

clean_pyc:
	find . -not \( -path './venv' -prune \) -name '*.pyc' -exec rm -f {} \;
