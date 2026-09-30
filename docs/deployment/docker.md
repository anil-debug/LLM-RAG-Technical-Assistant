# Docker Compose

This is one of three independent deployment paths:

```text
Docker Compose     local API, UI, and PostgreSQL + pgvector
Kubernetes         the same image on Minikube or another cluster
AWS EKS            the same image, with RDS instead of in-cluster Postgres
```

Compose does not install or require Kubernetes. Kubernetes does not read `docker-compose.yml`.

The Compose file builds one application image, `technical-rag-assistant:0.1.0`, and uses it for both the API and Streamlit. That is the same tag the Helm chart expects.

| Service | Image | Port | Role |
| --- | --- | --- | --- |
| `postgres` | `pgvector/pgvector:pg16` | 5432 | PostgreSQL 16 with the `vector` extension |
| `ollama` | `ollama/ollama:latest` | 11434 | Local OpenAI-compatible LLM. Chat stays 503 until a model is pulled |
| `api` | `technical-rag-assistant:0.1.0` | 8000 | FastAPI, `VECTOR_BACKEND=postgres` |
| `ui` | same image, Streamlit command | 8501 | Talks to `http://api:8000` |

The database user and password default to `rag` / `rag`. That is a local development value. Do not reuse it for RDS.

## Configuration

```bash
cp .env.example .env
```

`.env` is gitignored. Compose loads it with `env_file` and also uses it for `${VAR:-default}` substitution.

Inside the Compose network these values replace the host-oriented ones from `.env`:

| Variable | Inside Compose | Why |
| --- | --- | --- |
| `VECTOR_BACKEND` | `postgres` | The stack always includes pgvector |
| `DATABASE_URL` | `postgresql://rag:rag@postgres:5432/rag` | `localhost` would be the API container itself |
| `API_BASE_URL` on the UI | `http://api:8000` | The browser is not the caller; the UI container is |

`LLM_BASE_URL` in `.env` is for a process on the host (`uv run`). Compose does not copy it into the API container, because `localhost` inside the container is the API itself. The container uses `COMPOSE_LLM_BASE_URL`, which defaults to `http://ollama:11434/v1`.

Pull the model once after the Ollama container is up. The name must match `LLM_MODEL`:

```bash
docker-compose exec ollama ollama pull qwen2.5:3b
```

`/health` and `/ready` succeed before that pull. `/chat` returns 503 `model_unavailable` when the Ollama process is down, and a model-not-found error when the server is up but the model was not pulled. Set `COMPOSE_LLM_BASE_URL=http://host.docker.internal:11434/v1` only when Ollama is already running on the host.

Change the published ports in `.env` with `API_PORT`, `UI_PORT`, and `POSTGRES_PORT` if 8000, 8501, or 5432 are already taken.

## Startup

Either binary works. This machine has Compose v1 (`docker-compose`). The v2 plugin (`docker compose`) is not installed here.

```bash
cp .env.example .env
# optional: point LLM_BASE_URL at the host Ollama
docker-compose up -d --build
```

`make compose-up` does the same copy and start. The Makefile uses `docker compose` when that plugin exists, and `docker-compose` otherwise.

The API waits until Postgres is healthy. The UI waits until the API answers `/health`.

## Check

```bash
docker-compose ps
curl -sS http://localhost:8000/health
curl -sS http://localhost:8000/ready
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:8501
curl -sS http://localhost:8000/documents
docker-compose exec postgres psql -U rag -d rag -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

`/ready` reports `vector_backend` `postgres` after the database accepts connections. `GET /documents` creates the tables and `CREATE EXTENSION vector` on first use. An empty list is a successful call.

The first semantic search downloads `BAAI/bge-base-en-v1.5` into the container unless that download is blocked. Chat needs a server at `LLM_BASE_URL`.

## Shutdown

```bash
docker-compose stop          # keep the database volume
docker-compose down          # remove containers, keep the volume
docker-compose down -v       # also delete the pgvector data volume
```

`make compose-down` is `docker-compose down` and keeps the volume.

## Logs

```bash
docker-compose logs --tail=100 api
docker-compose logs --tail=100 postgres
docker-compose logs --tail=100 ui
```

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `unknown command: docker compose` | Use `docker-compose` (v1). The plugin is optional. |
| `KeyError: 'ContainerConfig'` on `up` after a config change | Compose 1.29 cannot recreate a container on Docker 29. `docker-compose rm -sf api ui` and then `docker-compose up -d`. Do not use `down -v` unless you mean to delete the database. |
| `Couldn't find env file: .env` | `cp .env.example .env` |
| Port already allocated | Set `API_PORT`, `UI_PORT`, or `POSTGRES_PORT` in `.env`, then `docker-compose up -d` |
| API stays unhealthy | `docker-compose logs api`. `/health` does not need the embedding weights. A crash is a settings or import error. |
| `/ready` returns 503 | Postgres is not accepting connections yet, or `DATABASE_URL` does not use the hostname `postgres`. |
| `vector` extension missing | Call `GET /documents` once, or `docker-compose exec postgres psql -U rag -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"`. The image is `pgvector/pgvector:pg16`, which ships the extension. |
| `/chat` returns 503 `model_unavailable` | The API cannot open `COMPOSE_LLM_BASE_URL`. Default is `http://ollama:11434/v1`. Check `docker-compose ps` and `docker-compose logs ollama`. Pull the model with `docker-compose exec ollama ollama pull qwen2.5:3b`. |
| Search returns 503 `model_unavailable` | The container cannot download the embedding weights. Fetch them on a machine with network access, or mount a Hugging Face cache. |
| UI cannot reach the API | `API_BASE_URL` for the `ui` service must stay `http://api:8000`. Do not point it at `localhost`. |
| Uploads disappear after `down` | Upload bytes live on the container filesystem under `/app/data`. The database volume keeps chunk rows. `down -v` deletes those rows too. |

Uploads are not a named volume. A container recreate without a volume loses the original files. The database still has the extracted text until the volume is removed.

## What was executed here

On 2026-09-30, `docker-compose up -d --build` built `technical-rag-assistant:0.1.0` from cache and started all three containers. Postgres and the API reported healthy.

| Check | Result |
| --- | --- |
| `GET /health` | `{"status":"ok"}` |
| `GET /ready` | `vector_backend` `postgres`, embedding model `BAAI/bge-base-en-v1.5` |
| `GET /documents` | `[]` then, after upload, the troubleshooting manual |
| `pg_extension` | `vector` |
| Streamlit `GET /` | HTTP 200. The page was not clicked through in a browser |
| `POST /documents` of `troubleshooting.md` | 200, 5 chunks, model `BAAI/bge-base-en-v1.5` downloaded inside the container |
| `POST /search` for `What does error 0x1F mean?` | 200. Top hits were the fan-stall chunk (semantic 0.703, BM25 2.15) and the reserved-code chunk (semantic 0.632, BM25 4.56). Fusion scores were tied at about 0.0325. Reranker was off |
| `POST /chat` | 503 `model_unavailable`. Nothing is listening on `host.docker.internal:11434` |

The stack was left running. Stop it with `docker-compose down`. `docker-compose down -v` also deletes the uploaded document.
