# Design trade-offs

These are the choices an interviewer will ask about. The alternatives are real options. The selection is the one this repository implements.

## Why PostgreSQL + pgvector?

The deployment store needs documents, chunks, embeddings, and conversation turns in one transactional place. pgvector keeps the embedding next to the chunk row that cites it. Filtering by `model_name` and joining back to `documents` is ordinary SQL.

| Store | What it does well | Why it is not the default here |
| --- | --- | --- |
| FAISS | Fast local ANN, including GPU indexes | A separate process from the document metadata. No transactions, no conversation table. |
| Qdrant | Purpose-built vector service with payload filters | Another service to operate. The corpus also needs BM25, which Qdrant does not replace by itself. |
| Milvus | Large-scale ANN | Heavy for a portfolio service whose first corpus is a handful of manuals. |
| Elasticsearch / OpenSearch | Strong lexical search, vectors as an add-on | BM25 is the interesting lexical baseline to implement and explain, not to hide behind `ts_rank` or a Lucene query. |

FAISS remains a reasonable index if retrieval must leave the database. This project keeps one system of record so a citation can be resolved to a chunk row.

HNSW is the ANN index. It is approximate. The memory store used in tests is exact brute-force cosine, which is the right check for a few dozen chunks.

## Why BM25?

Embedding models are trained to put paraphrases near each other. They are weaker on strings that must match exactly:

- `0x1F`
- `01.73.12`
- `SYS_FAN1`
- `POST code 0xD4`
- `FRU-FAN-220`

BM25 scores term frequency against a document-frequency penalty (Robertson–Sparck Jones IDF, `k1 = 1.5`, `b = 0.75`). A rare identifier that appears in one chunk outranks a chunk that only shares ordinary words such as "error" and "server".

The tokenizer keeps those strings intact. `0x1F`, `01.73.12`, and `SYS_FAN1` are each one token, then casefolded. A search for `sys_fan1` hits `SYS_FAN1`.

On this sample, BM25 alone is measurable without a neural model. The number belongs to `evaluation/reports/bm25_retrieval_*.json`. It is not an embedding result.

## Why hybrid retrieval?

Semantic search and BM25 fail in different places. A paraphrase of "how do I sign in" may never contain `X-Auth-Token`. An embedding of `0x1F` may sit near other short hex strings. Reciprocal rank fusion adds the two ranks without pretending the raw scores share a unit:

```text
RRF(d) = sum over lists  1 / (k + rank(d))
```

`k = 60`. A chunk that only BM25 found still gets `1 / (60 + rank)`. A chunk in both lists gets both terms and rises. Fusion is in `retrieval/hybrid_search.py`.

## Why rerank?

The bi-encoder encodes the query and the chunk separately. It never sees them in one sequence, so it cannot model token-level interaction. The cross-encoder (`BAAI/bge-reranker-base`) scores the pair together. That is more accurate and much more expensive, so it only sees the fused top 30 and returns 8. `RERANKER_ENABLED` defaults to false so the first boot does not download the cross-encoder.

## Why BGE as the default embedding model?

`BAAI/bge-base-en-v1.5` is a 768-dimensional English bi-encoder with a published retrieval instruction. The query prefix is `Represent this sentence for searching relevant passages: ` (the trailing space is part of the prefix). Passages are not prefixed. Vectors are L2-normalized here, even if the library can normalize them, so the normalization step is visible.

## Why compare with E5?

`intfloat/e5-base-v2` is also 768-dimensional, so it can share `vector(768)`. It uses different prefixes: `query: ` and `passage: `. A comparison that ignores prefixes is not a comparison of the models as they were trained. `embeddings/config.py` owns that difference. The quality comparison was not executed in this environment because the weights were not cached and Hugging Face could not be reached. See the embedding benchmark report.

## Why Ollama first?

Ollama speaks the OpenAI chat-completions API on localhost. That lets the RAG loop, prompts, and citations be developed without loading a multi-gigabyte model into this process. vLLM is the same HTTP client pointed at another base URL. The local Transformers provider exists so the forward pass can be inspected and, on a GPU, served without a separate daemon.

## Why a provider protocol?

Retrieval must not know whether the tokens came from Ollama, vLLM, or `model.generate`. `answer_question` calls `generate`. Tests pass a scripted `FakeLLM`. A benchmark can freeze the retrieved context and swap only the generator, which is the comparison that does not confound retrieval with the model.

## Why Transformers after the HTTP provider?

The HTTP provider does not show tokenization, attention, or the KV cache. `TransformersLocalProvider` and `scripts/inspect_model.py` do. The inspect script's default run is a randomly initialized tiny GPT-2. That execution is real and is not a quality score. A pretrained load is a separate command and was not executed here.

## Why LoRA instead of full fine-tuning?

Full fine-tuning updates every weight. For a 7B model that is tens of gigabytes of optimizer state. LoRA freezes `W` and learns `W' = W + BA`, with rank `r` much smaller than the hidden size (`r = 8`, `alpha = 16`, targets `q_proj` and `v_proj` in this plan). QLoRA additionally stores the frozen base in 4-bit (NF4 via bitsandbytes) and trains the adapters in higher precision. bitsandbytes is an optional extra and is not installed here. Training is skipped when `torch.cuda.is_available()` is false. A CPU fallback would be a different procedure, so it is not silently substituted.

The training questions are generated from the manuals and checked so none of them equal a golden question string. The golden set stays held out. The four-arm comparison (base, base+RAG, fine-tuned, fine-tuned+RAG) was not executed.

## Why Kubernetes?

The API, the UI, configuration, and secrets are separate objects. Readiness (`/ready`) gates traffic until the process can ping its store. Liveness (`/health`) only checks that the process answers. Resource requests and limits keep a model process from consuming a node without a bound. The chart renders with `helm template`. It was later installed on Minikube. That run is not an EKS deployment.

## Why EKS?

The target cloud is AWS. EKS is managed Kubernetes, which matches the Helm chart. RDS for PostgreSQL can run pgvector. S3 holds original uploads. An ALB ingress exposes the UI and the API on separate hostnames. Secrets Manager holds `DATABASE_URL` and the LLM credential. A GPU node group is optional and is not required for the API process when generation stays on Ollama or vLLM elsewhere. None of those resources were created. The design is in `docs/deployment/aws.md`.

## Chunking on a small corpus

The chunk-strategy report compares fixed, token, recursive, and structure-aware chunking with the same BM25 scorer. On these short manuals, the coarser strategies often emit one chunk per file. That inflates recall: the labeled fact is somewhere in the only chunk. Structure-aware chunking produces more, smaller chunks (headings preserved, logs split on events). Recall can dip and precision can rise because a hit is a tighter span. That is a property of the metric and the corpus size, not a reason to abandon structure. The default remains structure-aware, around 480 tokens with about 72 tokens of overlap, because a citation should name a section rather than an entire manual.

## Document intelligence

Entity patterns, linear classifiers, and summaries are independent flags and default off. The entity-boost ablation on BM25 is executed and the delta is small. Classifiers and summarization were not ablated against retrieval quality, because those runs need embedding weights and an LLM. The project does not claim that document intelligence improves answers.
