# Architecture

The assistant answers questions about server-platform documents. It retrieves evidence first, then asks a language model to write from that evidence, then checks the citation ids.

```text
Documents
  -> loaders + cleaning
  -> metadata
  -> optional document intelligence
  -> structure-aware chunking
  -> bi-encoder (BGE by default, E5 in the benchmark config)
  -> PostgreSQL + pgvector, or an in-memory cosine store
User question
  -> query rewrite (history is used only here)
  -> semantic search + BM25
  -> reciprocal rank fusion
  -> optional cross-encoder rerank (top 30 -> top 8)
  -> context builder with citation ids
  -> LLM provider (Ollama, vLLM, or local Transformers)
  -> answer + validated citations
```

LangChain and LlamaIndex are not used. Chunking, BM25, fusion, prompts, citations, and evaluation live in this repository. Libraries are used for the pieces that are not the point of the project: PDF parsing, HTML parsing, PyTorch, Transformers, sentence-transformers, FastAPI, and PEFT.

## Components

| Path | Responsibility |
| --- | --- |
| `ingestion/` | Load PDF, Markdown, text, HTML, and logs. Clean text. Assign document ids and checksums. Chunk. |
| `intelligence/` | Optional entities, section labels, document labels, and summaries. Each flag defaults to off. |
| `embeddings/` | Bi-encoder interface, BGE and E5 prefixes, L2 normalization. |
| `retrieval/` | Memory store, pgvector store, BM25, fusion, cross-encoder. |
| `generation/` | Provider protocol, prompts, rewrite, context packing, citation checks. |
| `evaluation/` | Golden questions, metric functions, JSON reports. |
| `api/` | FastAPI routes. |
| `frontend/` | Streamlit page that shows the intermediate RAG state. |
| `finetune/` | LoRA/QLoRA plan. Training does not start without CUDA. |

## Request path

1. `POST /documents` checks the suffix, size, and filename. The stored name is the basename, so `../../secret.md` becomes `secret.md`.
2. `prepare_document` loads, cleans, and chunks. Intelligence runs only when its flag is on.
3. The embedder encodes `embed_text` (section heading plus chunk text). Vectors are L2-normalized.
4. The store upserts the document, chunks, and vectors. The same checksum replaces the previous document.
5. `POST /chat` loads prior user and assistant turns. It does not put retrieved chunks into that history.
6. If history exists, the LLM rewrites the follow-up into one standalone query. Retrieval uses that string. The generator still sees the original history plus the new evidence.
7. Hybrid retrieval embeds the rewritten query, runs BM25 over every stored chunk, and fuses the two ranked lists with RRF (`k = 60`).
8. If `RERANKER_ENABLED=true`, `BAAI/bge-reranker-base` rescores the fused window and keeps 8 chunks.
9. `build_context` packs chunks until the token budget is spent and numbers them `[1]`, `[2]`, ...
10. The model answers. Bracket ids that were not packed are removed. A source list is appended from the chunks we actually stored.

## Two indexes, one column

`chunk_embeddings` is `vector(768)`. BGE and E5 are both 768-dimensional, so they share the column. Rows are keyed by `(chunk_id, model_name)`. Each model has its own partial HNSW index:

```sql
CREATE INDEX ... ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
WHERE model_name = 'BAAI/bge-base-en-v1.5';
```

The model name is interpolated only after it matches `[A-Za-z0-9_./:+-]+`. Cosine distance in pgvector is `<=>`. The returned score is `1 - distance`, which equals the dot product of the normalized vectors.

The memory store is the default so the API can boot without PostgreSQL. It is exact cosine search. It is not a quality model and it is not shared across processes.

## Providers

`LLMProvider.generate(messages, temperature, max_tokens)` is the only generation method the rest of the code calls.

- `OpenAICompatibleProvider` POSTs `{base}/v1/chat/completions`. Ollama is `http://localhost:11434/v1`. vLLM is the same client with a different `LLM_BASE_URL`.
- `TransformersLocalProvider` loads a causal LM with PyTorch. It calls `model.eval()` and generates inside `torch.inference_mode()`.

`resolve_device` returns `cpu` or `cuda`. A CUDA request on a machine without a GPU raises `ModelUnavailableError`. It does not silently run on CPU.

## Security boundary

Retrieved text is placed in the user message and labeled untrusted. The system message tells the model not to follow instructions inside the evidence. That is prompt-injection control, not a guarantee. Upload type, size, and path checks sit in `api/service.py`. `get_principal` is the authentication extension point. The default principal is anonymous.

## What is intentionally out of scope

Long-term memory, multi-tenant auth, and a learned NER tagger are not in this build. AWS resources are designed in `docs/deployment/aws.md` and were not created.
