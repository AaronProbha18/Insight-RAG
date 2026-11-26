# All generators read domains/**/*.yaml and enums/*.yaml. Nothing under build/ is edited by hand.
export PYTHONPATH := tools
PY ?= uv run python

.PHONY: all validate ddl explorer migrate check test clean

all: validate ddl explorer

validate:
	$(PY) tools/validate.py

ddl: validate
	$(PY) tools/ddl.py

explorer: validate
	$(PY) tools/explorer.py

# Writes migrations/NNNN_<name>.sql and advances model.lock.json. Usage: make migrate NAME=add_plan_description
migrate: validate
	$(PY) tools/migrate.py $(if $(NAME),--name $(NAME)) $(ARGS)

# CI gate: the model is valid and model.lock.json has no unmigrated changes.
check: validate
	$(PY) tools/migrate.py --dry-run | grep -q '^no changes' || (echo "model has unmigrated changes: run make migrate" && exit 1)

test:
	$(PY) -m pytest -q -rs tests

clean:
	rm -rf build
