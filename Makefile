.PHONY: install test lint router bench

install:
	uv sync --extra dev

_install_pip:
	pip install -e ".[dev]"

test:
	uv run pytest -q

lint:
	uv run ruff check manifold tests

router:
	MANIFOLD_BACKENDS?=http://localhost:8001,http://localhost:8002 \
	uv run python -m manifold.router

bench:
	uv run python -m manifold.loadgen --base-url http://localhost:8000 \
		--rate 4 --duration 120 --sessions 32 --out bench/results.csv
