# All generators read $(MODEL)/domains/**/*.yaml and $(MODEL)/enums/*.yaml. Nothing under build/ is edited by hand.
export PYTHONPATH := tools
PY ?= uv run python
MODEL ?= models/plumbline

.PHONY: all validate ddl explorer migrate test clean

all: validate ddl explorer

validate:
	$(PY) tools/validate.py --model $(MODEL)

ddl: validate
	$(PY) tools/ddl.py --model $(MODEL)

explorer: validate
	$(PY) tools/explorer.py --model $(MODEL)

# Writes migrations/NNNN_<name>.sql and advances model.lock.json. Usage: make migrate NAME=add_plan_description
migrate: validate
	$(PY) tools/migrate.py --model $(MODEL) $(if $(NAME),--name $(NAME)) $(ARGS)

test:
	$(PY) -m pytest -q -rs tests

clean:
	rm -rf $(MODEL)/build
