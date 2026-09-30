.PHONY: test evaluate api ui helm-template compose-config inspect train-lora

export UV_LINK_MODE ?= copy

test:
	uv run pytest

evaluate:
	uv run python scripts/evaluate.py bm25
	uv run python scripts/evaluate.py chunk-strategies
	uv run python scripts/evaluate.py entities
	uv run python scripts/evaluate.py entity-boost
	uv run python scripts/benchmark_models.py
	uv run python scripts/inspect_model.py
	uv run python scripts/train_lora.py
	uv run python scripts/compare_finetune.py
	uv run python scripts/train_classifiers.py

api:
	uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

ui:
	uv run streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501

helm-template:
	helm template technical-rag-assistant deployment/helm/technical-rag-assistant

compose-config:
	docker compose config

inspect:
	uv run python scripts/inspect_model.py

train-lora:
	uv run python scripts/train_lora.py
