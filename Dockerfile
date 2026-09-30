# Application image. The build downloads the CPU PyTorch wheel and is large.
# This repository does not claim the image was built unless `docker build` was run.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/
WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY core core
COPY ingestion ingestion
COPY intelligence intelligence
COPY embeddings embeddings
COPY retrieval retrieval
COPY generation generation
COPY evaluation evaluation
COPY api api
COPY finetune finetune
COPY frontend frontend
COPY data/sample data/sample

RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
