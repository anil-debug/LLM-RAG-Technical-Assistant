# Docker

`docker-compose.yml` defines three services:

| Service | Image | Role |
| --- | --- | --- |
| `postgres` | `pgvector/pgvector:pg16` | PostgreSQL 16 with the `vector` extension |
| `api` | Built from the repository `Dockerfile` | FastAPI on port 8000, `VECTOR_BACKEND=postgres` |
| `ui` | Same image, Streamlit command | Port 8501, `API_BASE_URL=http://api:8000` |

The database password `rag` is a local development value. Do not reuse it on AWS.

```bash
docker compose up --build
```

On this machine the Compose v2 plugin (`docker compose`) is not installed. Compose v1.29.2 is installed as `docker-compose`, and `docker-compose config` succeeded. The image was not built. `docker compose up` was not executed, so this document does not claim a running stack.

The API container reaches Ollama on the host through `LLM_BASE_URL` (default `http://host.docker.internal:11434/v1`). The first search downloads the embedding model into the container unless the cache is mounted.

The schema is created on first use by `PostgresVectorStore.ensure_schema`, including `CREATE EXTENSION vector` and a partial HNSW index per embedding model name.

Uploads in the container are written under `/app/data/documents`. That directory is inside the container filesystem. A restart without a volume loses uploads. The Kubernetes chart mounts an `emptyDir` for the same reason, and the AWS design moves originals to S3.
