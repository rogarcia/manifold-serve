.PHONY: install test lint router sim

MANIFOLD_BACKENDS ?= http://localhost:8001,http://localhost:8002
SIM_PORT ?= 8001

install:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check manifold tests

router:
	MANIFOLD_BACKENDS=$(MANIFOLD_BACKENDS) uv run python -m manifold.router

sim:
	uv run python -m manifold.sim --port $(SIM_PORT)
