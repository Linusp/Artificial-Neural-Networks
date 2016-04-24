VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PIP_COMPILER = $(VENV)/bin/pip-compile
PIP_MIRROR = https://pypi.mirrors.ustc.edu.cn/simple
PIP_INSTALL = $(PIP) install --exists-action=w

setup: venv deps

deps: venv
	@$(PIP_COMPILER) -i $(PIP_MIRROR) reqs.in -o reqs.txt
	@$(PIP_INSTALL) -i $(PIP_MIRROR) -r reqs.txt

venv:
	@virtualenv $(VENV) --prompt '<venv:connections>'
	@$(PIP_INSTALL) -i $(PIP_MIRROR) -U pip setuptools pip-tools

clean_pyc:
	find . -not \( -path './venv' -prune \) -name '*.pyc' -exec rm -f {} \;
