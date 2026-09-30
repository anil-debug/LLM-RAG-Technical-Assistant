#!/usr/bin/env bash
# Start the API. Chat still needs Ollama or another OpenAI-compatible server.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
exec uv run uvicorn api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"
