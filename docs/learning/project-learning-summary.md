# What this project teaches

Read this after you can run `uv run pytest` and open one BM25 report. Each step names the idea, the file, why it matters, and a question you should be able to answer without notes.

## Python

The service is ordinary Python: packages at the repository root, `pyproject.toml`, uv, and pytest. Settings are a `pydantic-settings` model (`core/settings.py`). Logging is the standard library (`core/logging_config.py`) with `request_id`, `query_id`, and `document_id` on every line. Errors are typed exceptions (`core/errors.py`) mapped to HTTP status codes at the edge.

Why it matters: an interviewer can ask you to trace one request without a framework hiding the call stack.

Question: where does an upload that is too large turn into HTTP 413?

## NLP

`core/text.py` tokenizes with a regex that prefers hex (`0x1F`) and dotted numbers (`01.73.12`) over a plain integer split. Tokens are casefolded. That vocabulary is for BM25 and chunk sizes. It is not a model vocabulary. Subword tokenization appears when a Transformers tokenizer is loaded. `scripts/inspect_model.py` shows integer ids going into a GPT-2 embedding.

Question: why must `SYS_FAN1` stay one BM25 token?

## Embeddings

`embeddings/embedding_model.py` defines `embed_documents` and `embed_query`. BGE prefixes only the query. E5 prefixes both sides (`embeddings/config.py`). `l2_normalize` makes cosine similarity a dot product (`embeddings/embed.py`). A zero vector becomes a one-hot so a blank string cannot produce NaN.

Question: why is the BGE prefix applied to the query and not the passage?

## Vector search

`MemoryVectorStore.search_semantic` dots the query with every stored vector and sorts. `PostgresVectorStore` uses `1 - (embedding <=> query)` and an HNSW index filtered by `model_name`. Both stores also keep the chunk text needed for a citation.

Question: why can two 768-d models share one `vector(768)` column?

## BM25

`retrieval/bm25.py` implements

```text
IDF(t) = log(1 + (N - n_t + 0.5) / (n_t + 0.5))
score = IDF * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))
```

with `k1 = 1.5` and `b = 0.75`. The executed golden-set numbers are in `evaluation/reports/bm25_retrieval_20260930_020512.json`.

Question: why does a rare identifier beat a chunk that only shares the word "error"?

## Hybrid retrieval

`reciprocal_rank_fusion` sums `1 / (60 + rank)` across the semantic list and the BM25 list. Scores are not added raw, because cosine and BM25 do not share a unit. `tests/test_rrf.py` checks that a keyword-only hit survives.

Question: what happens to a chunk that the bi-encoder missed and BM25 ranked first?

## Reranking

`CrossEncoderReranker.rerank` scores `(query, chunk)` pairs and stable-sorts them. The window is the fused top 30. The default flag is off. Tests inject `predict` so the suite never downloads `BAAI/bge-reranker-base`.

Question: why is a cross-encoder not run on every chunk in the corpus?

## RAG

`generation/answer.py` is the loop: rewrite, retrieve, pack, generate, resolve citations. `generation/context.py` numbers packed blocks from 1 and can append the next chunk when it shares a section and still fits the budget. The system prompt and the evidence live in different messages (`generation/prompt.py`).

Question: walk the path from a PDF upload to the `[1]` in the answer.

## LLMs

`LLMProvider` is a protocol with one method. `OpenAICompatibleProvider` is Ollama and vLLM. Tests use `FakeLLM` and a fake HTTP client. No Ollama process was running here, so no live generation metric exists.

Question: what changes in the code when you move from Ollama to vLLM?

## Transformers

`docs/design/transformers.md` is the equation-level note. The executed demonstration is a random GPT-2 of 22,272 parameters: logits shape `[1, 5, 128]`, attention shape `[1, 4, 5, 5]`, four new tokens, CPU, float32. Report: `evaluation/reports/transformers_random_init_20260930_020527.json`. Those ids are not language quality.

Question: what does dividing `QK^T` by `sqrt(d_k)` prevent?

## PyTorch

The linear head (`intelligence/linear_classifier.py`) is the smallest training loop in the repo: `model.train()`, `loss.backward()`, `optimizer.step()`, then `model.eval()` and `torch.inference_mode()` for prediction. `resolve_device` refuses a silent CUDA fallback.

Question: why is `inference_mode` wrong during LoRA training and right during serving?

## Evaluation

Recall@K is fact recall, not "the filename appeared." Precision uses denominator k. Unanswerable questions are not in the retrieval average. Generation proxies are named as proxies. Reports keep nulls.

Question: why would file-level recall flatter a chunker that returns the whole manual?

## Model benchmarking

`configs/benchmark.yaml` names BGE, E5, and two Ollama model ids. `scripts/benchmark_models.py` wrote `NOT_EXECUTED` because Hugging Face and Ollama were unreachable. The chunk-strategy report is a real comparison, and it compares chunkers under BM25, not embedding models.

Question: why must the LLM comparison freeze the retrieved context?

## LoRA / QLoRA

`finetune/train.py` builds `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])`. The update is `W' = W + BA`. The base stays frozen. QLoRA needs bitsandbytes, which is not installed. CUDA was not available, so the report is `NOT_EXECUTED` and no adapter directory was written. 240 training questions were written to `data/sample/labels/train.jsonl` and checked against the golden strings.

Question: what would you compare before saying the adapter helped?

## FastAPI

`api/main.py` builds the app, assigns request ids, and maps `AppError` subclasses to status codes. `tests/test_api.py` uploads, searches, chats, follows up, and rejects a bad type, an empty file, a huge file, and a path-like filename. The store and the LLM in that test are fakes.

Question: why is the stored filename `passwd.md` when the client sent `../../etc/passwd.md`?

## Docker

The Compose file starts pgvector, the API, and Streamlit from one image tag. On this machine `docker-compose up -d --build` ran that stack. `/chat` stayed 503 because Ollama was not running. The measured search hits are in `docs/deployment/docker.md`, not in the benchmark JSON.

Question: why does the API service set `VECTOR_BACKEND=postgres` inside Compose while the default on a laptop is `memory`?

## Kubernetes

The Helm chart renders a Deployment, Service, ConfigMap, Secret, an optional Postgres StatefulSet, probes, and resource limits. `helm template` succeeded, including the ingress and GPU overlays. The chart was also installed on Minikube, upgraded, and rolled back. EKS was not installed.

Question: what is the difference between `/health` and `/ready`?

## AWS

`docs/deployment/aws.md` places the chart on EKS, pgvector on RDS, files on S3, traffic on an ALB, and credentials in Secrets Manager. No account was used.

Question: which IAM actions does the API role need, and which does it not need?
