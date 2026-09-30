# Final verification report

Date of the commands below: 2026-09-30. Machine: Linux x86_64, 76 CPUs, 66,992,488,448 bytes MemTotal, Python 3.12.14, torch 2.14.0+cpu, `torch.cuda.is_available()` false. No `nvidia-smi`. Nothing was listening on `127.0.0.1:11434`.

`git_revision` inside the JSON reports is `unknown` because those files were written before this repository had a commit. That field is honest. It is not a hash.

## Phase status

| Phase | Status | What was checked |
| --- | --- | --- |
| 1 Environment | PASS | `uv sync`, settings, logging, request ids. `tests/test_config.py`, `tests/test_logging.py`. |
| 2 Ingestion | PASS | PDF, Markdown, text, and HTML loaders. `tests/test_ingestion.py`, `tests/test_cleaning.py`. |
| 3 Chunking | PASS | Fixed, token, recursive, structure, and log splits. `tests/test_chunking.py`. Strategy metrics executed. |
| 4 Embeddings and pgvector | PARTIAL | Interface, BGE/E5 prefixes, L2 normalization, and the pgvector schema are implemented. Neural encode was not run. The Postgres round trip is skipped unless `RUN_POSTGRES=1`. |
| 5 Semantic retrieval | PARTIAL | Implemented. Unit-tested with the hashing embedder, which is not a quality score. No BGE vectors were searched. |
| 6 BM25 and hybrid | PASS | BM25 and RRF are implemented and unit-tested. BM25 on the golden set was executed. Hybrid fusion was unit-tested. The neural half of hybrid was not executed against BGE. |
| 7 Reranker | PARTIAL | `CrossEncoderReranker` is implemented. Tests inject scores. `BAAI/bge-reranker-base` was not downloaded. The flag defaults off. |
| 8 LLM provider | PARTIAL | OpenAI-compatible client is implemented and tested with a fake HTTP response. Ollama was down, so no live completion was recorded. |
| 9 Citations | PASS | Ids resolve to chunk metadata. Unknown ids are rejected. `tests/test_citations.py`. |
| 10 Query rewrite | PASS | History rewrites a standalone query in unit tests with `FakeLLM`. A live rewriter was not called. |
| 11 Base evaluation | PARTIAL | 50 golden questions exist. BM25 Recall, Precision, and MRR were executed. Faithfulness, answer relevance, and citation accuracy were not executed against a model. |
| 12 Document intelligence | PARTIAL | Entity patterns were scored. The linear head was trained on synthetic features in a unit test. Real classifier training and summary ablation were not executed. Entity-boost ablation was executed. |
| 13 Model benchmark | PARTIAL | The benchmark script ran and wrote `NOT_EXECUTED` for embeddings and LLMs. The random-init Transformers inspect ran. |
| 14 FastAPI | PASS | Upload, list, get, delete, search, chat, health, ready, and validation. `tests/test_api.py`. |
| 15 Streamlit | PARTIAL | `frontend/app.py` and `frontend/view.py` exist. Hit rows are unit-tested. The page was not exercised in a browser. |
| 16 Transformers inference | PARTIAL | `TransformersLocalProvider` is implemented (`eval` and `inference_mode`). A random GPT-2 forward pass and `generate` ran on CPU. A pretrained checkpoint was not loaded. |
| 17 LoRA / QLoRA | SKIPPED | The training function, PEFT config, and 240-pair dataset exist. Training did not start. Reason: no CUDA. Report status `NOT_EXECUTED`. |
| 18 Docker | PASS | `docker-compose up -d --build` on 2026-09-30. `/health`, `/ready`, upload, pgvector, and hybrid search succeeded. `/chat` is 503 without Ollama. Streamlit returned HTTP 200 and was not opened in a browser. |
| 19 Kubernetes and Helm | PASS | Minikube install with in-cluster pgvector. Probes passed. `helm upgrade` and `helm rollback` succeeded. EKS was not used. |
| 20 AWS | SKIPPED | Design is written. No AWS API was called. |

PARTIAL means the code is there and the part that does not need extra services was tested. It does not mean a neural quality number exists.

## Commands and results

### Unit, API, and integration tests

```text
UV_LINK_MODE=copy uv run pytest --tb=line -q --disable-warnings
```

Result: 72 passed, 1 skipped. The skip is `tests/test_vector_store.py` unless `RUN_POSTGRES=1`. The suite does not call Ollama. `tests/test_integration.py` indexes the sample corpus with the hashing embedder and a scripted answer.

### BM25 and related reports

```text
uv run python scripts/evaluate.py bm25
uv run python scripts/evaluate.py chunk-strategies
uv run python scripts/evaluate.py entities
uv run python scripts/evaluate.py entity-boost
```

All four wrote `status: EXECUTED`. Quote numbers only from those JSON files. The README tables are rounded from:

- `evaluation/reports/bm25_retrieval_20260930_020512.json`
- `evaluation/reports/chunk_strategy_20260930_020513.json`
- `evaluation/reports/entity_extraction_20260930_020515.json`
- `evaluation/reports/intelligence_ablation_20260930_020517.json`

### Benchmarks

```text
uv run python scripts/benchmark_models.py
```

Embedding report: `NOT_EXECUTED`. BGE and E5 were not in the cache, and the process could not reach huggingface.co. LLM report: `NOT_EXECUTED`. Both configured base URLs returned connection refused.

### Transformers inspect

```text
uv run python scripts/inspect_model.py
```

`EXECUTED` for `random-gpt2-config`. 22,272 parameters, CPU, float32, logits `[1, 5, 128]`, attention `[1, 4, 5, 5]`, 4 new tokens, `cuda_peak_bytes` null. This is not a pretrained model and not a quality score.

### Fine-tuning

```text
uv run python scripts/train_lora.py
uv run python scripts/compare_finetune.py
uv run python scripts/train_classifiers.py
```

All three wrote `NOT_EXECUTED`. `train_lora.py` also wrote 240 training pairs to `data/sample/labels/train.jsonl`. No directory `models/lora` was created.

### Docker

```text
docker-compose -f docker-compose.yml config
```

Docker 29.1.3. Compose v1.29.2. The `docker compose` plugin is not installed on this host. `docker-compose up -d --build` later tagged `technical-rag-assistant:0.1.0` and started postgres, api, and ui. See `docs/deployment/docker.md`.

### Helm

```text
helm template technical-rag-assistant deployment/helm/technical-rag-assistant
helm template technical-rag-assistant deployment/helm/technical-rag-assistant \
  --set ingress.enabled=true --set gpu.enabled=true
```

Both succeeded. The GPU render contains `nvidia.com/gpu`, `/health`, and `/ready`. A later Minikube install, upgrade, and rollback are recorded in `docs/deployment/kubernetes.md`. That install is not an EKS deployment.

## Checklist

| Item | Result |
| --- | --- |
| Unit tests | 72 passed |
| Integration tests | Passed inside the same pytest run (`tests/test_integration.py`) |
| API tests | Passed inside the same pytest run (`tests/test_api.py`) |
| Docker validation | Stack started. `/health`, `/ready`, upload, and search passed. Chat 503 without Ollama. |
| Helm validation | `helm template` passed. Minikube install, upgrade, and rollback passed. EKS not installed. |
| Model availability | BGE, E5, and the reranker were not loaded |
| GPU availability | None. `nvidia-smi` missing. CUDA false |
| Ollama availability | Not running |
| AWS execution | Not executed |
| Fine-tuning execution | Not executed. Requires a GPU |
| Benchmark execution | Embedding and LLM quality benchmarks not executed. BM25, chunk strategy, entities, entity boost, and the random-init forward pass were executed |

## How to finish the incomplete phases

Neural retrieval and the embedding benchmark, after network access to the model weights:

```bash
uv run python scripts/benchmark_models.py --download
uv run python scripts/ingest.py
```

LLM metrics, after Ollama or vLLM is serving:

```bash
# example only: ollama serve, then pull a model that fits the machine
uv run python scripts/benchmark_models.py
```

The script currently records reachability. Scoring frozen contexts is described in `docs/user-guide/evaluation.md` and is not filled in with guesses.

Postgres integration:

```bash
RUN_POSTGRES=1 DATABASE_URL=postgresql://rag:rag@localhost:5432/rag uv run pytest tests/test_vector_store.py
```

Pretrained Transformers inspect:

```bash
uv run python scripts/inspect_model.py --model <causal-lm-id>
```

LoRA, on a CUDA machine:

```bash
uv run python scripts/train_lora.py --model <causal-lm-id> --output models/lora
uv run python scripts/compare_finetune.py
```

QLoRA also needs `uv sync --extra qlora`.

Docker, on a host with the Compose v2 plugin if you want that exact command:

```bash
docker compose up --build
```

`docker-compose up -d --build` was run on this host. The result is in `docs/deployment/docker.md`.

Minikube was installed with `values-minikube.yaml`. EKS still needs a registry and a cluster:

```bash
helm upgrade --install technical-rag-assistant deployment/helm/technical-rag-assistant \
  -f deployment/helm/technical-rag-assistant/values-eks.yaml \
  --set image.repository=<account>.dkr.ecr.<region>.amazonaws.com/technical-rag-assistant \
  --set secret.databaseUrl='postgresql://USER:PASSWORD@RDS_HOST:5432/rag'
```

AWS: follow `docs/deployment/aws.md`. Nothing in that document has been applied.
