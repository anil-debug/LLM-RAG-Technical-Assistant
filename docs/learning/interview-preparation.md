# Interview preparation

Answers refer to this repository. If a number is quoted, it comes from a JSON report named in the answer. Generation quality, embedding Recall@K, fine-tuning, and AWS were not measured.

## Python

1. **Why is configuration a `Settings` object?** Callers read `settings.embedding_model` instead of calling `os.environ` in every module. `core/settings.py` validates dimension and chunk overlap at startup.

2. **What does `extra="ignore"` do?** An unknown variable in `.env` does not crash the process. A typo in a known name still falls through to the default, so important keys are tested in `tests/test_config.py`.

3. **Why do tests pass `_env_file=None`?** So a developer's `.env` cannot change a unit test. The API process uses `Settings()` and does read `.env`.

4. **How does a request id get onto a log line?** Middleware calls `set_request_id`. `RequestIdFilter` copies the context variable onto the log record. The format string prints `request_id`, `query_id`, and `document_id`.

5. **Why a `ContextVar` instead of a global?** Concurrent requests on one event loop must not share one id. A context variable follows the task.

6. **Why custom exceptions?** `DocumentNotFoundError` and `UploadTooLargeError` are mapped once in `api/main.py`. Routes do not pick status codes themselves.

7. **Where does HTTP 413 come from?** `save_upload` raises `UploadTooLargeError` when `len(data)` exceeds `max_upload_bytes`. The handler emits 413 `payload_too_large`.

8. **Why are packages at the repository root?** The import names are `ingestion`, `retrieval`, and `generation`. There is no `src/` layout. Hatch lists those directories in `pyproject.toml`.

9. **What does `uv sync` do?** It creates `.venv` from `uv.lock` and installs this project in editable mode. `UV_LINK_MODE=copy` avoids hardlink warnings when the home directory and the repo are on different filesystems.

10. **Why is torch pinned to the CPU index?** `[tool.uv.sources]` sends `torch` to `https://download.pytorch.org/whl/cpu`. This machine has no NVIDIA driver. A CUDA wheel would not make `torch.cuda.is_available()` true here.

11. **What is `LLMProvider`?** A `Protocol` with `generate`. `OpenAICompatibleProvider` and `TransformersLocalProvider` both satisfy it. `answer_question` type-checks against the protocol and does not import a server SDK.

12. **Why dataclasses for `Document` and `Chunk`?** They are records with no behavior beyond `embed_text` being filled by the chunker. Pydantic is used at the HTTP boundary, where validation matters.

13. **How do tests avoid a live LLM?** `tests/fakes.py` has `FakeLLM` and `HashingEmbedder`. The hashing embedder is a bag of tokens. Its scores are not retrieval quality.

14. **What does `log_event` add?** It appends `key=value` fields such as `embedding_ms` and `top_k` so a log search does not depend on a JSON parser.

15. **How is the version exposed?** `core/__init__.py` reads `importlib.metadata` for `technical-rag-assistant`. FastAPI uses that version.

16. **How do you add a setting?** Add a typed field on `Settings`, document it in `.env.example`, and cover the default or the validator in `tests/test_config.py`.

17. **What if `EMBEDDING_DIM` is 384 for BGE?** The validator raises. `KNOWN_EMBEDDING_DIMS` says BGE and E5 are 768. A silent mismatch would write vectors the `vector(768)` column cannot store.

18. **Why check upload size in the app?** The API must behave the same under uvicorn, Docker, and a future ALB. The limit is `max_upload_bytes`.

19. **What is `Container`?** `api/deps.py` builds the store, embedder, LLM, reranker, and token counter once. Tests pass their own container into `create_app` and never touch the global app's models.

20. **Where are secrets?** `.env` is gitignored. `.env.example` has local placeholders (`LLM_API_KEY=ollama`, database user `rag`). The Helm secret is a placeholder string, not a production password.

## NLP and embeddings

1. **What is a token in the BM25 tokenizer?** A match of `TOKEN_RE`: a hex literal, a dotted number, an identifier, or digits. The match is casefolded.

2. **Why casefold?** `sys_fan1` and `SYS_FAN1` are the same technical name. BM25 would otherwise miss one of them.

3. **Why does the regex try hex before plain digits?** `0x1F` must not become `0` plus `x1F`. Order in the alternation is the priority.

4. **How is that different from a model tokenizer?** BPE splits by a learned vocabulary. `SYS_FAN1` may be several subwords. BM25 wants the identifier whole. The two tokenizers are used for different jobs.

5. **What is a vocabulary?** The set of strings the tokenizer can emit, or, for a neural model, the set of ids in the embedding matrix. The random GPT-2 in the inspect script has `vocab_size` 128.

6. **What is an embedding?** A vector that places similar texts near each other. This project uses 768-d bi-encoder vectors.

7. **Why cosine similarity?** It ignores vector length and compares direction. After L2 normalization, cosine is the dot product, which is what both stores compute.

8. **Why normalize in our code?** `SentenceTransformerEmbedder` calls `encode(..., normalize_embeddings=False)` and then `l2_normalize`. The step is visible and shared with the memory store.

9. **What is the BGE query prefix?** `Represent this sentence for searching relevant passages: ` including the trailing space. Passages have an empty prefix. E5 uses `query: ` and `passage: `.

10. **Why do prefixes matter in a bake-off?** Each model was trained with its own prefix. Omitting it changes the vector. `preset_config` in `embeddings/config.py` is the bake-off setup. The bake-off itself was not executed.

11. **What does the bi-encoder return for a chunk?** One pooled vector, stored under `chunk_embeddings` with the model name. It does not return tokens to the user.

12. **Why are error codes a weak spot for embeddings?** The model may treat `0x1F` and `0x1A` as nearby short codes. BM25 treats them as different rare terms. The troubleshooting guide says `0x1A` is not the fan fault.

13. **What does overlap do?** Token windows repeat about 72 tokens so a fact on a boundary appears in two chunks. Structure-aware chunking still uses that overlap inside a long section.

14. **Why put the heading in `embed_text`?** The vector sees "Factory reset" plus the paragraph, so a query about factory reset is closer to that section than to a page that only shares the word BMC.

15. **Why a gazetteer before a learned NER model?** Versions, POST codes, and product names are regular. The labeled miss `ARX-77` shows the limit. A learned tagger is justified after those misses are measured, which is what the entity report does.

## RAG

1. **Walk me through your RAG system.** Upload, clean, chunk, embed, store. At question time: rewrite if there is history, hybrid retrieve, optional rerank, pack a budgeted context with ids, generate, drop unknown citation ids, append sources from chunk metadata.

2. **What is ingestion?** `prepare_document` loads PDF, Markdown, text, HTML, or a log, cleans it, extracts a title, and chunks it. Intelligence is optional and off by default.

3. **What is cleaning?** NFKC, dehyphenation of a word split across a newline, collapsed whitespace outside fences, and repeated page headers dropped when they repeat on most pages.

4. **How is a document identified?** UUID plus SHA-256 of the original bytes. The same checksum replaces the previous row so a re-upload does not duplicate chunks.

5. **Why structure-aware chunking as the default?** A citation should name a section. Logs split on timestamped events so two SEL lines are not one chunk. The target is about 480 tokens with about 72 of overlap.

6. **What did the chunker comparison show?** On this short corpus, fixed, token, and recursive chunking produced 9 chunks and BM25 Recall@5 of 1.0. Structure-aware produced 32 chunks, Recall@5 of about 0.953, and higher Precision@5. Coarse chunks inflate recall because the whole file is one hit. Source: `chunk_strategy_20260930_020513.json`.

7. **What is in the index?** Document row, chunk text, heading, pages, neighbor ids, and a 768-d vector per embedding model.

8. **What is semantic search?** Embed the query with the query prefix, compare to stored passage vectors, return the top k with cosine scores.

9. **What is BM25 doing in the same request?** It scores the raw query against chunk text. Exact identifiers do not depend on the bi-encoder.

10. **What is RRF?** `sum 1/(60+rank)` over the two lists. A chunk only BM25 found still receives a term.

11. **What is the rerank window?** Fused candidates, capped at `retrieval_candidates` (30), then the cross-encoder keeps `rerank_top_n` (8).

12. **How is context built?** Hits are packed in order until `context_token_budget`. The first block is kept even if it alone exceeds the budget. Citation numbers start at 1 in pack order, which may differ from the model's preferred order.

13. **How do citations work?** The model may write `[1]`. `resolve_citations` keeps ids that were packed and records the rest as rejected. `render_answer` strips rejected markers and replaces any model-written "Sources:" tail with one built from filename, section, page, and chunk id.

14. **What is untrusted evidence?** The chunk is data in the user message. The system message says not to follow instructions inside it. `tests/test_generation.py` checks that an "ignore previous instructions" string does not enter the system message.

15. **What is the abstention sentence?** `The provided documents do not contain enough information to answer this question.` Unanswerable golden questions expect it.

16. **How does multi-turn work?** History is prior user and assistant text. The rewriter turns "What about authentication?" into a standalone query. Retrieval uses that query. The generator still sees the history plus new evidence. History is not long-term memory.

17. **What is a citation failure?** An id that was not packed, or a citation whose file is not in the expected set. `citation_accuracy` requires no rejected ids.

18. **Why not put chunks into the chat history?** The next retrieval would search a mixture of old evidence and the new question. Evidence belongs only in the current user message.

19. **What does the UI show?** Rewritten query, each hit's semantic, BM25, fusion, and rerank scores, the packed context, the answer, the citations, and the timings. Helpers are in `frontend/view.py`.

20. **What did you measure end to end?** BM25 retrieval on 43 answerable questions. Recall@5 0.953, Recall@10 0.977, MRR 0.875, Precision@5 0.233, Precision@10 0.121. Mean BM25 search time about 0.093 ms. Process RSS about 62.6 MB. File: `bm25_retrieval_20260930_020512.json`. No LLM answer score was measured.

21. **Which questions are exact identifiers?** Fifteen, including `0x1F`, `01.73.12`, `NET-4401`, `AR-NVS-3.2.0`. Their BM25 Recall@5 was 1.0 and MRR was 1.0 in that report.

22. **Which questions must the system refuse?** Seven: default BIOS password, IPMI cipher, warranty, chassis serial, LDAP, SNMP community, live fan RPM. The manuals say those values are absent.

23. **What is document intelligence in the path?** Optional entity tags, a linear section label, a linear document label, and a summary chunk. Flags default off.

24. **Did entity boost help?** Recall@10 moved from about 0.977 to 1.0 and MRR from 0.875 to 0.878. Precision@5 did not change. That is a small move on 43 questions, not a reason to turn the flag on by default. File: `intelligence_ablation_20260930_020517.json`.

25. **What would you change first if recall fell on a larger corpus?** Stop rebuilding BM25 from every chunk on each query, and stop using one chunk per short file as the only unit. Keep structure-aware chunks so citations stay specific.

## Transformers

1. **What is a token embedding?** A row in a matrix of shape `(vocab, d_model)`, selected by the token id.

2. **What is positional information?** A vector added to the token embedding so the block can tell position 1 from position 5. The inspect script's GPT-2 config has `n_positions` 32.

3. **What are Q, K, and V?** Learned projections of the hidden states. Queries are compared to keys. Values are what get mixed.

4. **Write the attention equation.** `softmax(Q K^T / sqrt(d_k)) V`.

5. **Why divide by `sqrt(d_k)`?** Dot products grow with dimension. Large logits make softmax nearly one-hot and the gradient tiny.

6. **What is multi-head attention?** Several attention maps with smaller `d_k`, concatenated, then mixed by `W_O`. The random model used 4 heads. Attention shape was `[1, 4, 5, 5]`: batch, heads, query positions, key positions.

7. **What is a residual connection?** `x + block(x)`. The block learns a change. The original stream still flows forward.

8. **What is layer normalization?** It rescales activations so depth does not explode or vanish. Pre-norm blocks normalize before attention and before the feed-forward layer.

9. **What is the feed-forward network?** A per-position MLP. Attention mixes tokens. The MLP mixes features inside one token.

10. **Encoder versus decoder?** BGE and E5 are encoders. They read the whole chunk and pool one vector. The chat model is a decoder. It predicts the next token and cannot attend to future tokens.

11. **What is causal masking?** Scores for keys after the current position are set to a large negative value before the softmax. Position `t` only sees `0..t`.

12. **Bi-encoder versus cross-encoder?** The bi-encoder embeds query and chunk separately. The cross-encoder consumes the pair in one sequence. The first is the index. The second is the reranker.

13. **What shape are the logits?** `(batch, sequence, vocab)`. The inspect run recorded `[1, 5, 128]` for a 5-token prompt and a 128-word vocabulary.

14. **How does generation use those logits?** The last position is the next-token distribution. Greedy decoding takes argmax. The random model appended 4 tokens. That sequence is not English. The report says so.

15. **What is prefill?** The first forward pass over the prompt. A long RAG context makes prefill expensive.

16. **What is the KV cache?** Stored keys and values from earlier positions. Decode reuses them. `model.generate` owns the cache in this project.

17. **Why `model.eval()`?** Dropout turns off. The same prompt does not take a random subnetwork.

18. **Why `torch.inference_mode()`?** Autograd does not store activations. Serving does not need a backward pass. LoRA training does, so training must not use inference mode.

19. **How many parameters did the inspect model have?** 22,272. It is a tiny random GPT-2, dtype float32, device CPU. `cuda_peak_bytes` is null. RSS was 525,824,000 bytes for the process, which includes the already-imported libraries, not a GPU allocator.

20. **Why not quote that run as model quality?** The weights are the default initialization. No tokenizer was loaded. `generated_ids` repeat the last input id. The script exists to show shapes and the generation call.

## PyTorch

1. **What is a tensor?** A typed multidimensional array. `input_ids` in the inspect script is `torch.long` of shape `[1, 5]`.

2. **What is `nn.Linear`?** `y = x W^T + b`. The section and document heads are one linear layer. The embedding that feeds them is frozen.

3. **What is a parameter?** A tensor registered on a module and updated by the optimizer, unless `requires_grad` is false. LoRA freezes the base parameters.

4. **What does `loss.backward()` do?** It fills `.grad` on parameters that require grad, by the chain rule from the cross-entropy.

5. **What does `optimizer.step()` do?** It applies those gradients. The linear head uses Adam. The LoRA plan uses AdamW on parameters that require grad.

6. **What does `optimizer.zero_grad()` do?** It clears gradients so they do not accumulate across steps.

7. **Why `model.train()` during the head fit?** Dropout and batch-norm behave differently in train mode. This head has neither, but the call is still the right training state.

8. **Why seed the linear-head test?** `torch.manual_seed` makes the synthetic fit repeatable. The test expects every training row to be classified correctly because the two clouds are far apart. That is not manual accuracy.

9. **What is `torch.inference_mode()` versus `no_grad()`?** Both skip the backward graph. Inference mode is stricter and is what serving uses. The code calls inference mode.

10. **How do you move a model to a device?** `model.to(device)` after `resolve_device`. CUDA is requested only when `torch.cuda.is_available()` is true. Otherwise the call raises.

11. **What dtype would you use on a GPU?** float16 or bfloat16 for weights, often with a higher-precision reduction. The CPU inspect run stayed float32. `TransformersLocalProvider` accepts `float32`, `float16`, and `bfloat16`.

12. **How is memory measured?** Process RSS from `/proc/self/status` VmRSS. CUDA peak from `torch.cuda.max_memory_allocated` when CUDA exists. The method string is stored in the report. A null CUDA field means no GPU measurement.

13. **What is a state dict?** The tensors of a module. `load_section_classifier` loads one with `weights_only=True` and `map_location="cpu"`.

14. **Why `weights_only=True`?** It refuses to unpickle arbitrary objects from a checkpoint file.

15. **What runs on GPU if you had one?** The embedding model, the reranker, and `TransformersLocalProvider`. BM25 stays on CPU. The chart requests a GPU only when `gpu.enabled` is true.

## LLM inference

1. **What is a provider?** The object that turns chat messages into text. Retrieval does not know which one it is.

2. **What HTTP call does Ollama get?** `POST {LLM_BASE_URL}/chat/completions` with `model`, `messages`, `temperature`, and `max_tokens`. Default base URL `http://localhost:11434/v1`.

3. **How is vLLM selected?** Set `LLM_BASE_URL` to the vLLM server. The Python class does not change.

4. **What happens if the server is down?** `ModelUnavailableError`, HTTP 503. This environment got connection refused on port 11434. The LLM benchmark report records that and no scores.

5. **What is temperature 0 here?** Greedy decoding. The Transformers provider sets `do_sample` false and passes `temperature=None`, because a sampler temperature of 0 is invalid.

6. **What is `max_tokens`?** The cap on new tokens, not the prompt length. The context budget is separate and limits retrieved text before the call.

7. **What is latency in this design?** `rewrite_ms`, `retrieval_ms` (and inside it `embedding_ms`, `bm25_ms`, `reranker_ms`), `llm_ms`, and `total_ms`. They are logged with the model name and `top_k`.

8. **Prefill versus decode?** Prefill reads the prompt, including the evidence. Decode emits one token at a time using the KV cache.

9. **What dominates RAG latency?** Usually prefill on a long context, then decode, unless retrieval is loading a model on first use. BM25 on 32 chunks was about 0.09 ms and is not the bottleneck.

10. **What is batching?** Several prompts in one forward pass. This service generates one chat at a time. A vLLM server batches across requests. The provider does not implement batching itself.

11. **Throughput versus latency?** Latency is one user's wait. Throughput is tokens per second across users. A GPU increases throughput when requests are batched. One extra user on a saturated GPU increases latency.

12. **CPU versus GPU?** CPU runs the tiny inspect model and can run a small embedder slowly. A 7B chat model in 16-bit wants a GPU. This machine reported `cuda_available: false`.

13. **What is quantization?** Storing weights in fewer bits. QLoRA uses 4-bit NF4 for the frozen base. The `qlora` extra is not installed. No quantized model was loaded.

14. **What memory should you report for Ollama?** Client latency, plus process RSS only if you can see the local Ollama PID. A remote server gets client latency only. The benchmark config states those three methods.

15. **Why freeze context in an LLM bake-off?** Otherwise a better answer might be a luckier retrieval. The comparison is the generator. That bake-off was not run.

## Evaluation

1. **What is Recall@K here?** The fraction of labeled fact strings found in the top k hits whose file is expected. It is not "did the filename appear."

2. **Why refuse file-level recall?** One chunk that is the whole manual would count as a hit for every fact in that file, even when the cited span is the wrong section.

3. **What is Precision@K?** Relevant hits in the top k, divided by k. Missing slots count against you.

4. **Why is precision low even when recall is high?** k is 5 or 10, and most of those slots are not the labeled fact. Structure-aware Precision@5 was about 0.233 while Recall@5 was about 0.953.

5. **What is MRR?** One over the rank of the first relevant hit, averaged over questions. Exact-identifier MRR was 1.0. Multi-turn standalone questions were 0.629.

6. **How many questions?** 50 in `golden.jsonl`. 43 scored for retrieval. 7 unanswerable questions are excluded from retrieval and reserved for abstention.

7. **What is a multi-hop item?** A question whose labeled facts span more than one string, sometimes two files, such as the GPU bundle and the 70 C limit.

8. **How is an unanswerable question scored?** Relevance proxy is 1 only if the abstention sentence is present. Faithfulness proxy also requires no citations. Retrieval does not average these rows.

9. **What is the faithfulness proxy?** A labeled fact counts only if it is in the answer and in a cited chunk. It is not a general entailment model. The methodology string says it is not an LLM judge.

10. **What is citation accuracy?** Every cited file was expected, at least one citation exists for an answerable question, and no id was rejected. Unanswerable items must cite nothing.

11. **Why are judge fields null?** No judge model was called. Filling them would be a fabricated score.

12. **What is in a report?** Status, reason, git revision, timestamp, machine, Python, model name, configuration, dataset version, methodology, metrics. `git_revision` is `unknown` on the current files because they were written before the first commit.

13. **What did entity extraction score?** Micro precision 1.0 and recall about 0.909 on the hand-labeled passages. The only false negative is tracking code `ARX-77`. File: `entity_extraction_20260930_020515.json`.

14. **What must a later embedding report contain?** Recall@5, Recall@10, MRR, index time, query latency, and memory, for BGE and E5 separately, with the prefix each model requires.

15. **What must you not do with the hashing embedder?** Quote it as Recall@K. It only makes unit tests move vectors when tokens overlap.

## System design

1. **How is the system split?** Ingestion, optional intelligence, embeddings, retrieval, generation, API, UI. Each package can be tested without the others.

2. **Where is state?** PostgreSQL when `VECTOR_BACKEND=postgres`: documents, chunks, embeddings, conversations, messages. Memory is the single-process default.

3. **Why conversations in the same database?** A follow-up needs the prior turns, and those turns must not be the retrieval corpus. They are a different table.

4. **What is the read path's bottleneck at larger scale?** Rebuilding BM25 from `all_chunks()` on every query, then an ANN query. The UI is not the bottleneck.

5. **How do you scale the API?** Stateless replicas behind a Service. The index is in RDS, so replicas share it. BM25 state is still rebuilt per process unless it moves into the database.

6. **How do you scale generation?** A separate vLLM or Transformers deployment on a GPU node group. The API keeps `OpenAICompatibleProvider` and a base URL. CPU API replicas do not load the chat weights.

7. **What is the cache story?** Embedding vectors are computed at index time. The KV cache is inside one generation call. There is no cross-request answer cache, because citations must match the current index.

8. **How do you update a document?** Upsert on checksum. Old chunks for that checksum are deleted. HNSW rows go with the chunk via `ON DELETE CASCADE`.

9. **What is the failure mode if pgvector is down?** `/ready` returns 503. `/health` can still return 200. Liveness must not kill a pod just because the database blipped if you only want a restart when the process is dead. Readiness removes it from traffic.

10. **How are uploads validated?** Suffix allow-list, non-empty body, max bytes, basename only.

11. **What is the auth story?** `get_principal` returns anonymous. It is the extension point. The design does not pretend to have multi-tenant auth.

12. **How do you observe one failed chat?** Search logs for the `x-request-id`. The line includes query id, model name, top k, and the stage timings.

13. **What is stored for a citation?** Chunk id, document id, filename, title, section, page span. The source line is rendered from those fields.

14. **Why a token budget?** The model's context is finite and prefill cost grows with it. `context_token_budget` defaults to 3000 regex tokens.

15. **What happens on prompt injection in a manual?** The text stays in the user message. The system message forbids following it. This is a control, not a proof. A malicious manual can still bias a weak model. Citations let a human check.

16. **How would you add OpenSearch later?** As the BM25 engine, with pgvector or the same Postgres still holding chunks. Fusion stays in `hybrid_search.py`. The lexical list would come from OpenSearch instead of `BM25Index`.

17. **What is not in the critical path?** Section classifiers, document classifiers, summaries, LoRA, the Streamlit process. The API can answer with all intelligence flags off and the reranker off.

18. **How do you test the critical path?** `tests/test_integration.py` indexes the sample, retrieves `0x1F`, and checks that a scripted answer's `[1]` resolves to a real chunk. `tests/test_api.py` does the same over HTTP.

19. **What is the consistency story?** One transaction per upsert in Postgres. The memory store uses a lock. Readers on another API replica see the commit.

20. **What would you add before a public URL?** Authentication on `get_principal`, a real secret for the database, a size limit at the load balancer, and a decision about whether anonymous users may upload.

## Docker and Kubernetes

1. **What does the Dockerfile install?** Python 3.12, uv, and the locked dependencies, including the CPU torch wheel. The command is `uvicorn api.main:app`.

2. **Was the image built?** No. `docker-compose config` succeeded. `docker build` was not run.

3. **Why three Compose services?** Postgres with pgvector, the API, and Streamlit. The UI only speaks HTTP to the API.

4. **Why `depends_on` with a health check?** The API creates tables on first use. `pg_isready` avoids a crash on a half-started database. This is not a Kubernetes readiness probe.

5. **What is the local database password?** `rag`. It is a Compose development value. The AWS design uses Secrets Manager instead.

6. **How does the container reach Ollama?** `LLM_BASE_URL` defaults to `http://host.docker.internal:11434/v1`. Ollama was not running on this host.

7. **What is a Deployment?** A desired replica count and a pod template. The chart has one for the API and one for the UI.

8. **What is a Service?** A stable virtual IP and DNS name for those pods. The UI's `API_BASE_URL` is `http://<release>-api:8000`.

9. **What is a ConfigMap?** Non-secret settings: model id, base URL, `RERANKER_ENABLED`, log level. Changing it requires a restart to re-read the environment.

10. **What is a Secret?** `DATABASE_URL` and `LLM_API_KEY`. The chart value is a placeholder. A real install overrides it.

11. **What is liveness?** `GET /health`. It means the process can answer. It does not check the database.

12. **What is readiness?** `GET /ready`. It pings the store. A 503 removes the pod from the Service endpoints.

13. **Why both?** A database outage should stop traffic without restart-looping the pod. A dead process should be restarted.

14. **What resource limits are set?** API request 250m CPU and 1Gi memory, limit 2 CPU and 4Gi. UI request 100m and 256Mi, limit 500m and 1Gi. These are starting points, not measurements of the embedding model's peak RSS.

15. **What happens if the embedding model needs more than 4Gi?** The kubelet kills the container. You raise the limit after measuring RSS, you do not guess a production number from the BM25 report's 62 MB.

16. **How does the GPU flag work?** `gpu.enabled=true` adds `nvidia.com/gpu` to the API container limits and can set a node selector and tolerations. `helm template` with that flag succeeded. No pod was scheduled.

17. **Why is the GPU limit dangerous on the CPU API?** Every API replica would request a GPU. A real layout is a separate inference Deployment. The AWS note says that.

18. **What did `helm template` prove?** The chart renders, including probes, the secret, the config, and the optional ingress. It did not prove a cluster install.

19. **How is ingress routed?** UI on `rag.example.com`, API on `api.rag.example.com`. There is no `/api` path prefix, because FastAPI is mounted at `/`.

20. **What is `emptyDir` for?** `/app/data` so an upload has a writable directory. It disappears with the pod. Originals belong in S3 in the AWS design.

## AWS

1. **Was this deployed to AWS?** No.

2. **Why EKS?** The unit of deployment is already a Helm chart. EKS runs that chart without operating the control plane.

3. **Why RDS and not Postgres on a pod?** Backups, failover, and a disk that outlives the API pod. pgvector is available on RDS PostgreSQL. The extension is created by `ensure_schema`.

4. **Why S3?** Original files are blobs. The database stores the checksum and the text. A pod disk is the wrong system of record.

5. **Why an ALB?** The ingress class in the chart is `alb`. One load balancer, two host rules, TLS at the balancer.

6. **Why Secrets Manager?** `DATABASE_URL` and the LLM key should not be in git or in a baked image. The chart consumes a Kubernetes secret that a controller fills from Secrets Manager.

7. **What is the network shape?** Public subnets for the ALB. Private subnets for nodes and RDS. No public IP on the database. Node security group is the only RDS client.

8. **What IAM does the API role get?** `s3:GetObject` and `s3:PutObject` on one prefix, `secretsmanager:GetSecretValue` on two ARNs. It does not get `rds:*` or `s3:*` on all buckets.

9. **What IAM does the node role get?** The standard EKS node permissions. It is not the application role. Pods use IRSA.

10. **Why IRSA?** The pod assumes a role. You do not put long-lived keys in the secret if the AWS API can be called with a web identity token. The database password is still a secret, because Postgres does not speak IRSA.

11. **How do you rotate the database password?** Rotate it in Secrets Manager, roll the Kubernetes secret, restart the API. The application reads the environment at process start.

12. **What is multi-AZ for?** RDS failover. It does not make HNSW faster. It is an availability choice. A lab can be single-AZ. The design says so and does not pick a price.

13. **How do embeddings survive a failover?** They are rows in RDS. They fail over with the database. They are not on the node disk.

14. **How would you scale read traffic?** More API replicas, RDS read replicas only if the ANN query is the bottleneck and you accept replication lag. BM25 rebuilt in each pod does not use a read replica until the postings live in SQL.

15. **How would you scale the GPU?** A separate node group, tainted, scale to zero, tolerated only by the inference deployment. The CPU API does not request `nvidia.com/gpu`.

16. **What do you log in CloudWatch?** The same stdout lines: request id, stage timings, model name. ALB 5xx and RDS CPU sit beside them.

17. **What must you not call GPU memory?** Client time waiting on a remote vLLM. The report methodology has to name the meter.

18. **Where does the Helm image come from?** ECR, after a `docker build` and `docker push` that were not run. The chart's `image.repository` is overridden at install.

19. **What is the install command you would run later?** `helm upgrade --install` with `vectorBackend=postgres`, a real `databaseUrl`, and `ingress.enabled=true`. It is written in `docs/deployment/aws.md`. It was not executed.

20. **What is the first security review item?** Replace the placeholder secret, restrict the ALB to the right CIDR, and confirm the API role cannot read other secrets.

## Why X instead of Y?

1. **Why PostgreSQL + pgvector instead of FAISS?** Citations need the chunk row, the filename, and the conversation in one commit. FAISS is an index, not that system of record. FAISS is still reasonable if ANN must leave the database.

2. **Why not Qdrant or Milvus?** They are good vector services. This corpus also needs documents, BM25, and chat turns. One Postgres database is the smaller operational surface. The trade-off is HNSW inside Postgres rather than a dedicated ANN cluster.

3. **Why not Elasticsearch for BM25?** The point of the project is to show the BM25 formula and the identifier tokenizer. Hiding them in Lucene would make the interview answer "the search engine did it."

4. **Why BM25 and embeddings together?** Embeddings retrieve paraphrases such as "how do I sign in." BM25 retrieves `0x1F` and `01.73.12`. Each misses the other's cases.

5. **Why RRF instead of adding the raw scores?** Cosine is roughly in `[-1, 1]`. BM25 is an unbounded weighted sum. Adding them lets one scale drown the other. Ranks are comparable.

6. **Why k = 60?** It is the usual RRF constant from the fusion literature used in this code. A smaller k rewards the top rank more sharply. The value is `rrf_k` and can be changed without rewriting the formula.

7. **Why a cross-encoder instead of a bigger bi-encoder?** The bi-encoder cannot look at query and chunk tokens together. The cross-encoder can, and it is only affordable on a short list. The default is off until the weights are present.

8. **Why BGE instead of E5 as the default?** Both are 768-d English bi-encoders. BGE is the default because its query instruction is the one the first index uses. E5 is the paired benchmark, with its own prefixes. Neither quality number was measured here.

9. **Why not a 384-d model?** The column is `vector(768)`. A 384-d model needs another table. The validator rejects a dimension that disagrees with BGE or E5.

10. **Why Ollama before vLLM and Transformers?** The RAG loop can be built against a local HTTP server. vLLM is a base-URL change. Transformers is there when you need the forward pass, not as the first dependency.

11. **Why a provider instead of calling httpx in `answer_question`?** Tests, Ollama, vLLM, and a local model share `generate`. The citation logic stays one place.

12. **Why Transformers at all if Ollama exists?** Ollama does not show token ids, attention shapes, `eval`, or `inference_mode`. The inspect script does.

13. **Why LoRA instead of full fine-tuning?** Full fine-tuning updates every weight and the optimizer state. LoRA trains `A` and `B` with the base frozen: `W' = W + BA`. QLoRA stores that base in 4-bit. Neither was executed.

14. **Why not fine-tune before measuring RAG?** A fine-tuned model can sound fluent and still invent `01.73.12`. RAG is the mechanism that binds the answer to a chunk. The four-arm comparison is how you would show that an adapter added something RAG did not.

15. **Why structure-aware chunks if coarser chunks scored higher recall?** Recall on a one-chunk manual is an easy metric. A citation of an entire file is a weak citation. Precision rose when sections were split. The default stays structure-aware.

16. **Why not LangChain?** The retrieval, fusion, prompts, and citations are the project. A framework that hides them removes the thing the interview is about.

17. **Why FastAPI and Streamlit?** FastAPI is the contract tests hit. Streamlit is a window onto rewritten queries and scores. The UI does not embed the corpus itself.

18. **Why Kubernetes instead of only Compose?** Compose is the laptop. Kubernetes adds probes, limits, a service account, and a secret object that match how EKS would run the same image. Helm is how those objects are parameterized.

19. **Why EKS instead of running Postgres on the same node?** RDS outlives the pod and has backups. The node disk is scratch. The chart's `emptyDir` is explicitly not the system of record.

20. **Why leave authentication as a hook?** The sample is a single-user technical demo. Inventing a login system would not teach RAG. `get_principal` is where a real check would land without rewriting routes.

## Questions the reports do not answer

Do not invent these if they are asked. Say they were not measured and name the command.

- Which embedding model has higher Recall@10 on this corpus? Not measured. `uv run python scripts/benchmark_models.py --download`
- Does the reranker raise MRR? Not measured. Set `RERANKER_ENABLED=true` after the weights download.
- What is answer faithfulness? Not measured. Start Ollama and extend the LLM benchmark to score frozen contexts.
- Did LoRA beat RAG? Not measured. CUDA was unavailable. `uv run python scripts/train_lora.py --model <id>`
- What did AWS cost? Not estimated. No pricing API was called.
