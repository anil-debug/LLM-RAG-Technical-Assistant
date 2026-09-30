# Quick start

Requirements: Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/). This machine used uv 0.12.21 and CPython 3.12.14. If `uv sync` warns about hardlinks across filesystems, `UV_LINK_MODE=copy` avoids the warning.

```bash
uv python install 3.12
UV_LINK_MODE=copy uv sync
uv run pytest
```

One Postgres test is skipped unless `RUN_POSTGRES=1` and a pgvector database is reachable.

Copy `.env.example` to `.env` when you want Ollama or PostgreSQL. The file is gitignored. Tests build `Settings(_env_file=None)` so a developer `.env` does not change them.

## API and UI

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
uv run streamlit run frontend/app.py
```

`GET /health` answers even when the models are not downloaded. The first semantic search loads `BAAI/bge-base-en-v1.5`. Chat calls `LLM_BASE_URL` (default `http://localhost:11434/v1`). If Ollama is not running, chat returns 503 `model_unavailable`.

Index the sample manuals into a running API only after the embedding weights can be downloaded:

```bash
uv run python scripts/ingest.py
```

`VECTOR_BACKEND=memory` keeps the index in that process. Use `postgres` when the index must survive the process. `--skip-embed` stores chunks for BM25 and does not download a model. Semantic search then returns nothing.

## Evaluations that run here

```bash
uv run python scripts/evaluate.py bm25
uv run python scripts/evaluate.py chunk-strategies
uv run python scripts/evaluate.py entities
uv run python scripts/evaluate.py entity-boost
uv run python scripts/inspect_model.py
uv run python scripts/benchmark_models.py
uv run python scripts/train_lora.py
```

`benchmark_models.py` does not download weights unless you pass `--download`. `train_lora.py` writes the training JSONL and a `NOT_EXECUTED` report when CUDA is absent.

## Sample corpus

`data/sample/manuals/` and `data/sample/logs/sel.log` are original AstraRack X11 notes written for this project. They are not DMTF or vendor manuals. The golden questions are `evaluation/datasets/golden.jsonl` (`sample-v1`). Training questions are generated from `finetune/facts.py` and must not copy a golden question string.
