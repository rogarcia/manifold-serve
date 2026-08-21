.PHONY: install test lint router bench

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check manifold tests

router:
	MANIFOLD_BACKENDS?=http://localhost:8001,http://localhost:8002 \
	python -m manifold.router

bench:
	python -m manifold.loadgen --base-url http://localhost:8000 \
		--rate 4 --duration 120 --sessions 32 --out bench/results.csv
