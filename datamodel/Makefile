# All generators read $(MODEL)/domains/**/*.yaml and $(MODEL)/enums/*.yaml. Nothing under build/ is edited by hand.
export PYTHONPATH := tools
PY ?= uv run python
MODEL ?= models/plumbline

.PHONY: all validate ddl explorer test clean

all: validate ddl explorer

validate:
	$(PY) tools/validate.py --model $(MODEL)

ddl: validate
	$(PY) tools/ddl.py --model $(MODEL)

explorer: validate
	$(PY) tools/explorer.py --model $(MODEL)

test:
	$(PY) -m pytest -q -rs tests

clean:
	rm -rf $(MODEL)/build
