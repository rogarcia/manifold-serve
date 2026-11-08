# One image for the router, the sim and the autoscaler. Build from the repository root:
#
#   docker build -f deploy/k8s/Dockerfile -t manifold-serve:dev .
FROM python:3.12-slim-bookworm

ENV UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv==0.13.7

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY manifold ./manifold
RUN uv sync --frozen --no-dev --no-editable --extra hf

ENV PATH=/app/.venv/bin:$PATH HF_HOME=/tmp/huggingface
USER 65532:65532
CMD ["python", "-m", "manifold.router"]
