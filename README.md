# Technical RAG Assistant

A technical-document assistant for server-platform manuals. Ingestion, chunking, BM25, reciprocal rank fusion, citations, and evaluation are implemented in this repository. LangChain and LlamaIndex are not used.

The corpus is original. The manuals describe a fictional AstraRack X11 platform written for this project. They are not vendor or DMTF documents.

## Execution status

Measured results are copied from `evaluation/reports/`. A blank or null metric was not run. `git_revision` in those files is `unknown` because the reports were generated before the first commit. Re-run the evaluate scripts after clone if you want the revision stamped.

| Area | State | Evidence |
| --- | --- | --- |
| Unit and API tests | Executed locally | `uv run pytest`: 72 passed, 1 skipped (`RUN_POSTGRES` is unset) |
| Ingestion, chunking, BM25, RRF, citations, rewrite | Implemented and unit-tested | `tests/` |
| BM25 retrieval on the golden set | Executed locally | `evaluation/reports/bm25_retrieval_20260930_020512.json` |
| Chunk-strategy comparison | Executed locally | `evaluation/reports/chunk_strategy_20260930_020513.json` |
| Entity extraction | Executed locally | `evaluation/reports/entity_extraction_20260930_020515.json` |
| Entity-boost ablation | Executed locally | `evaluation/reports/intelligence_ablation_20260930_020517.json` |
| Tiny random GPT-2 forward pass | Executed locally on CPU | `evaluation/reports/transformers_random_init_20260930_020527.json` |
| BGE vs E5 quality | Not executed | The comparison report was not re-run. A later Compose upload did download BGE. E5 was not scored |
| LLM answer quality | Not executed | Ollama was not listening on port 11434 |
| Reranker model weights | Not loaded | The class is tested with an injected scoring function |
| LoRA / QLoRA training | Not executed | `torch.cuda.is_available()` is false |
| Classifier training on real embeddings | Not executed | Embedding weights were not available |
| Docker Compose | Executed locally | `docker-compose up -d --build` on 2026-09-30. `/health`, `/ready`, upload, and hybrid search succeeded. `/chat` returned 503 because Ollama was not running |
| Kubernetes | Executed on Minikube | Helm install, upgrade, and rollback. Pods ready, pgvector extension created. EKS was not used |
| AWS | Not executed | Design only, in `docs/deployment/aws.md` |
| Streamlit in a browser | Not executed | View-model helpers are unit-tested. No browser session was run |

## Why this project exists

The goal is a system you can walk through in an interview: where a chunk comes from, why an identifier is a BM25 problem, what a cross-encoder sees that a bi-encoder does not, how a decoder turns logits into the next token, and how you would run the same image on EKS without pretending the cluster already exists.

## Key capabilities

- Loaders for PDF, Markdown, text, HTML, and logs, with checksums and page or section metadata.
- Fixed, token, recursive, and structure-aware chunking. Structure-aware is the default (about 480 tokens, about 72 of overlap).
- Bi-encoder interface for `BAAI/bge-base-en-v1.5`, with `intfloat/e5-base-v2` as the paired benchmark configuration. Both are 768-d.
- PostgreSQL 16 + pgvector (cosine, partial HNSW per model name) and an in-memory cosine store for tests.
- BM25 (Robertson–Sparck Jones IDF, `k1=1.5`, `b=0.75`) and reciprocal rank fusion (`k=60`).
- Optional cross-encoder rerank (`BAAI/bge-reranker-base`), top 30 to top 8.
- LLM providers: OpenAI-compatible HTTP (Ollama or vLLM) and a local Transformers generator.
- Citation ids resolved to document, page, section, and chunk. Unknown ids are rejected.
- Conversational query rewrite that does not use history as the retrieval string.
- Optional document intelligence, each flag defaulting to off.
- FastAPI and a Streamlit page that shows the intermediate scores.
- Docker Compose, Kubernetes manifests, and a Helm chart.
- An AWS design for EKS, RDS + pgvector, S3, ALB, and Secrets Manager.

## Architecture

```text
Documents
  -> loaders + cleaning
  -> metadata
  -> optional document intelligence
  -> structure-aware chunking
  -> bi-encoder
  -> PostgreSQL + pgvector, or the in-memory store
User question
  -> query rewrite
  -> semantic search + BM25
  -> reciprocal rank fusion
  -> optional cross-encoder
  -> context with citation ids
  -> LLM provider
  -> answer + checked citations
```

Detail and the trade-offs against FAISS, Qdrant, Milvus, and Elasticsearch are in `docs/design/architecture.md` and `docs/design/tradeoffs.md`.

## Technology stack

Python 3.11+ (development used 3.12.14), uv, pytest, pydantic-settings, standard-library logging, PyTorch, Transformers, sentence-transformers, PEFT, FastAPI, Streamlit, PostgreSQL 16, pgvector, Docker, Kubernetes, Helm. The AWS design is documented and was not applied.

## Repository structure

```text
core/            settings, logging, errors, tokenizer, device selection
ingestion/       loaders, cleaning, chunking, metadata
intelligence/    entities, linear heads, summaries
embeddings/      bi-encoder adapter and prefixes
retrieval/       vector stores, BM25, fusion, reranker
generation/      providers, prompts, rewrite, citations
evaluation/      metrics, golden set, JSON reports
api/             FastAPI
frontend/        Streamlit
finetune/        LoRA plan and disjoint training questions
data/sample/     synthetic manuals, SEL log, labels
deployment/      Kubernetes manifests and the Helm chart
docs/            design, user guide, deployment, learning
scripts/         ingest, evaluate, inspect, train
```

## Quick start

```bash
uv python install 3.12
UV_LINK_MODE=copy uv sync
uv run pytest
```

Copy `.env.example` to `.env` before pointing the API at Ollama or PostgreSQL. `.env` is gitignored.

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
uv run streamlit run frontend/app.py
```

The first semantic search loads `BAAI/bge-base-en-v1.5` from Hugging Face. Chat requires an OpenAI-compatible server at `LLM_BASE_URL`. Neither download nor Ollama was available when this README's status table was written.

More commands: `docs/user-guide/quickstart.md`.

## Example usage

Upload:

```bash
curl -s -H 'x-request-id: demo-1' \
  -F 'file=@data/sample/manuals/troubleshooting.md' \
  http://localhost:8000/documents
```

Search:

```bash
curl -s http://localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"What does error 0x1F mean?","top_k":5}'
```

Chat:

```bash
curl -s http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"What does error 0x1F mean?"}'
```

A follow-up sends the returned `conversation_id`. The response includes `rewritten_query`, hit scores, `context`, `citations`, `rejected_citation_ids`, and timings. The API tests exercise this with a scripted model, not with Ollama. See `docs/user-guide/api.md`.

## RAG pipeline

```text
ingestion -> chunking -> embedding -> retrieval -> reranking -> context -> generation -> citation
```

Retrieved chunks are untrusted evidence in the user message. The system message tells the model not to follow instructions inside them. If the evidence is insufficient, the model is told to answer with exactly:

```text
The provided documents do not contain enough information to answer this question.
```

`generation/answer.py` is the function to read first.

## Document intelligence

| Flag | Default | What it does |
| --- | --- | --- |
| `INTEL_ENTITIES` | false | Regex and gazetteer for versions, POST codes, FRUs, endpoints |
| `INTEL_SECTION_CLASSIFIER` | false | Linear layer on a frozen chunk embedding |
| `INTEL_DOCUMENT_CLASSIFIER` | false | Linear layer on a frozen document embedding |
| `INTEL_SUMMARIZE` | false | LLM summary appended as an optional chunk |

`ARX-77` is a tracking code in the firmware guide. The patterns do not match it. The entity report's only false negative is that label. Micro precision on the hand-labeled passages was 1.0 and micro recall was 0.909 (10 true positives, 1 false negative). Adding the entity bonus to BM25 changed Recall@10 from 0.977 to 1.0 and MRR from 0.875 to 0.878. That is not treated as a reason to enable the flag by default. Classifiers and summaries were not ablated. See `docs/design/document-intelligence.md`.

## Evaluation

Fifty golden questions in `evaluation/datasets/golden.jsonl` (`sample-v1`): 15 exact-identifier, 15 technical, 8 multi-hop, 7 unanswerable, 5 multi-turn.

Recall@K is fact recall. A hit counts when the labeled substring appears in a top-k chunk from an expected file. Precision@K uses denominator k. Unanswerable questions are not in the retrieval average. Generation metrics in code are deterministic proxies and were not run against a live model.

### BM25, structure-aware chunks, 32 chunks, 43 scored questions

From `evaluation/reports/bm25_retrieval_20260930_020512.json`:

| Metric | Value |
| --- | --- |
| Recall@5 | 0.9535 |
| Recall@10 | 0.9767 |
| Precision@5 | 0.2326 |
| Precision@10 | 0.1209 |
| MRR | 0.8754 |
| Mean BM25 search latency | 0.0925 ms (`time.perf_counter` around `BM25Index.search` only) |
| Process RSS | 62,554,112 bytes (`/proc/self/status` VmRSS, not GPU memory) |

Exact-identifier Recall@5 and MRR were 1.0. Multi-turn standalone questions were the weak slice (Recall@5 0.60, MRR 0.629).

### Chunk strategies, same BM25 scorer

From `evaluation/reports/chunk_strategy_20260930_020513.json`:

| Strategy | Chunks | Recall@5 | Recall@10 | MRR | Precision@5 |
| --- | --- | --- | --- | --- | --- |
| fixed | 9 | 1.0 | 1.0 | 0.9186 | 0.2047 |
| token | 9 | 1.0 | 1.0 | 0.9186 | 0.2047 |
| recursive | 9 | 1.0 | 1.0 | 0.9186 | 0.2047 |
| structure | 32 | 0.9535 | 0.9767 | 0.8754 | 0.2326 |

On these short manuals the coarser strategies often emit one chunk per file, which makes file-level facts easy to "retrieve" and harder to cite precisely. Structure-aware chunking stays the default.

## Benchmark results

| Comparison | Status | Report |
| --- | --- | --- |
| BGE vs E5 Recall@K, latency, memory | Not executed | `embedding_benchmark_20260930_020522.json` |
| Two instruct LLMs, frozen contexts | Not executed | `llm_benchmark_20260930_020522.json` |
| Random tiny GPT-2 shapes | Executed, not a quality score | `transformers_random_init_20260930_020527.json` |

The random model had 22,272 parameters, dtype float32, device CPU, logits shape `[1, 5, 128]`, attention shape `[1, 4, 5, 5]`, and 4 new tokens. `cuda_peak_bytes` is null. Process RSS was 525,824,000 bytes after importing PyTorch. Do not read `generated_ids` as text quality.

To run the real embedding comparison after the weights can be downloaded:

```bash
uv run python scripts/benchmark_models.py --download
```

## Transformers

Token ids become vectors. Positional vectors are added. Each block runs attention and a feed-forward layer with residual connections and layer normalization:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

BGE and E5 are encoders. They pool a chunk into one vector. The chat model is a decoder. It predicts the next token and uses a causal mask. The cross-encoder reranker reads the query and the chunk in one sequence. The bi-encoder never does.

`model.eval()` disables dropout. `torch.inference_mode()` disables autograd for serving. Training must not use inference mode.

The full note is `docs/design/transformers.md`. The learning path is `docs/learning/project-learning-summary.md`.

## Fine-tuning

LoRA learns a low-rank update on a frozen base:

```text
W' = W + B A
```

The plan uses rank 8, alpha 16, and target modules `q_proj` and `v_proj`. QLoRA would load that base in 4-bit and is an optional extra (`bitsandbytes`), which is not installed. Training wrote 240 questions to `data/sample/labels/train.jsonl`. None of those strings are golden questions. No adapter was trained. The four-way comparison (base, base+RAG, fine-tuned, fine-tuned+RAG) is `NOT_EXECUTED`.

```bash
uv run python scripts/train_lora.py --model <causal-lm-id> --output models/lora
```

That command exits without training on a machine where CUDA is unavailable. Explanation: `docs/design/prompting-rag-finetuning.md`.

## Choose a deployment path

These three paths do not depend on each other. They share the image tag `technical-rag-assistant:0.1.0` and the same environment variable names.

```text
Docker Compose     one machine: API, Streamlit, PostgreSQL + pgvector
Kubernetes         Minikube or another cluster, including in-cluster Postgres
AWS EKS            the same Helm chart, RDS instead of the StatefulSet
```

Commands, shutdown, upgrade, rollback, and the results of the local runs are in `docs/deployment/docker.md`, `docs/deployment/kubernetes.md`, and `docs/deployment/aws.md`.

## Docker

```bash
cp .env.example .env
docker-compose up -d --build
```

On this host the v2 plugin is absent, so the command is `docker-compose`. `make compose-up` picks whichever binary exists. The database password `rag` is for this local stack only.

The stack was started here. `/health` and `/ready` succeeded, a manual was indexed with `BAAI/bge-base-en-v1.5`, and hybrid search returned the `0x1F` / `SYS_FAN1` section. `/chat` returned 503 because Ollama was not running. Streamlit answered HTTP 200 and was not exercised in a browser.

## Kubernetes

```bash
helm upgrade --install technical-rag-assistant \
  deployment/helm/technical-rag-assistant \
  -f deployment/helm/technical-rag-assistant/values-minikube.yaml
```

The default chart runs API, UI, and PostgreSQL + pgvector. `values-eks.yaml` turns that database off and expects an RDS URL. Raw manifests are `kubectl apply -k deployment/kubernetes`.

On this host, Minikube v1.35.0 ran the chart. The three pods became ready, `GET /documents` created the `vector` extension, `helm upgrade` recorded revision 2, and `helm rollback` returned to revision 1. EKS was not used.

## AWS

The target is EKS for the chart, RDS PostgreSQL with pgvector, S3 for original files, an ALB, Secrets Manager, and an optional GPU node group. IAM for the API is limited to one S3 prefix and two secrets. No AWS API was called. Install with `values-eks.yaml` only after the image is in a registry the nodes can pull. See `docs/deployment/aws.md`.

## Security

- `.env` is gitignored. No production secret is committed. Compose and Helm passwords are local placeholders.
- Uploads are limited by type, size, and basename. `../../etc/passwd.md` is stored as `passwd.md`.
- Retrieved text is untrusted evidence, not a system message.
- `get_principal` is the authentication hook. The default principal is anonymous.
- CUDA is not silently replaced with CPU.
- Classifier checkpoints load with `weights_only=True`.
- The AWS design uses IRSA and a narrow API role. It was not applied.

## Limitations

- Neural retrieval, reranking, and LLM answers were not measured here.
- BM25 is rebuilt from all chunks on every query. That is fine for 32 chunks and is the first scale limit.
- The memory store dies with the process.
- Coarse chunkers look better on Recall@K on this small corpus because a whole manual is one hit.
- Entity patterns miss tracking codes such as `ARX-77` on purpose.
- Streamlit was not clicked through in a browser.
- Docker Compose and Minikube were started on this machine. AWS was not deployed. Streamlit was not clicked through in a browser. Chat needs a running Ollama or vLLM.
- LoRA was not trained.

## Future improvements

- Run the BGE versus E5 benchmark and the frozen-context LLM benchmark, and commit only the JSON those commands write.
- Train the linear heads on real embeddings and ablate them. Keep the result even if it does not help.
- Move BM25 postings into Postgres or OpenSearch once the chunk count leaves the sample range.
- Split GPU inference into its own Deployment.
- Put originals in S3 and stop treating `emptyDir` as storage.
- Replace `get_principal` when the API is reachable beyond one user.

## Learning outcomes

The intended path is Python, NLP, embeddings, vector search, BM25, hybrid retrieval, reranking, RAG, LLMs, Transformers, PyTorch, evaluation, benchmarking, LoRA, FastAPI, Docker, Kubernetes, and AWS. Each step is tied to a file and an interview question in `docs/learning/project-learning-summary.md`. Practice questions are in `docs/learning/interview-preparation.md`.

Phase-by-phase status is in `docs/verification/FINAL_VERIFICATION_REPORT.md`.
