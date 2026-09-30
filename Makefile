.PHONY: test evaluate api ui helm-template compose-config compose-up compose-down compose-logs inspect train-lora

# Prefer the v2 plugin. Fall back to the v1 binary this host has.
COMPOSE := $(shell if docker compose version >/dev/null 2>&1; then echo docker compose; else echo docker-compose; fi)

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
	helm template technical-rag-assistant deployment/helm/technical-rag-assistant -f deployment/helm/technical-rag-assistant/values-minikube.yaml
	helm template technical-rag-assistant deployment/helm/technical-rag-assistant -f deployment/helm/technical-rag-assistant/values-eks.yaml
	helm template technical-rag-assistant deployment/helm/technical-rag-assistant --set ingress.enabled=true --set gpu.enabled=true

compose-config:
	test -f .env || cp .env.example .env
	$(COMPOSE) -f docker-compose.yml config

compose-up:
	test -f .env || cp .env.example .env
	$(COMPOSE) -f docker-compose.yml up -d --build

compose-down:
	$(COMPOSE) -f docker-compose.yml down

compose-logs:
	$(COMPOSE) -f docker-compose.yml logs --tail=100

inspect:
	uv run python scripts/inspect_model.py

train-lora:
	uv run python scripts/train_lora.py
