# HTTP API

The process is `api.main:app`. Errors use `{"error": {"code": "...", "message": "..."}}`. Every response includes `x-request-id`. Send one to correlate logs, or let the middleware generate it.

| Method | Path | Success | Failure |
| --- | --- | --- | --- |
| `POST` | `/documents` | 200 `DocumentCreated` | 415 bad type, 413 too large, 422 empty |
| `GET` | `/documents` | 200 list | |
| `GET` | `/documents/{id}` | 200 document plus chunks | 404 |
| `DELETE` | `/documents/{id}` | 200 | 404 |
| `POST` | `/search` | 200 hits and `retrieval_ms` | 422 if the query is empty |
| `POST` | `/chat` | 200 answer, citations, timings | 503 if the LLM or embedder cannot be used |
| `GET` | `/health` | 200 `ok` | |
| `GET` | `/ready` | 200 when the store pings | 503 when PostgreSQL is down |

`get_principal` runs on the document and chat routes and returns an anonymous principal. Replace that function to enforce a credential. Routes already depend on it.

## Upload

```bash
curl -s -H 'x-request-id: demo-1' \
  -F 'file=@data/sample/manuals/troubleshooting.md' \
  http://localhost:8000/documents
```

The filename is the basename only. A path is not stored.

## Search

```bash
curl -s http://localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"What does error 0x1F mean?","top_k":5}'
```

Each hit can carry `semantic_score`, `bm25_score`, `fusion_score`, and `rerank_score`. A stage that did not run leaves its score null.

## Chat

```bash
curl -s http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"What does error 0x1F mean?"}'
```

Send the returned `conversation_id` on the next turn. The service stores the user and assistant text. It does not store retrieved chunks as history. The response includes `rewritten_query`, `context`, `citations`, `rejected_citation_ids`, and timings: `rewrite_ms`, `retrieval_ms`, `llm_ms`, `total_ms`.

Logs from hybrid retrieval add `embedding_ms`, `bm25_ms`, and `reranker_ms`, plus `model_name` and `top_k`. The line format is timestamp, level, logger name, `request_id`, `query_id`, `document_id`, and the message.

## Example citation

A valid answer ends with a source block the server built:

```text
[1] Troubleshooting — Errors — page 1 — troubleshooting.md — chunk <id>
```

An id the model invented is listed in `rejected_citation_ids` and removed from the body.
