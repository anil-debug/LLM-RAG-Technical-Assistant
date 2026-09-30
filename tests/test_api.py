"""HTTP API with an in-memory store and a scripted LLM."""

from frontend.view import citation_rows, hit_rows

from fastapi.testclient import TestClient

from api.deps import Container
from api.main import create_app
from core.text import RegexTokenCounter
from retrieval.vector_store import MemoryVectorStore
from tests.fakes import FakeLLM, HashingEmbedder


def _reply(messages: list[dict[str, str]]) -> str:
    last = messages[-1]["content"]
    if "standalone search query" in last:
        return "How does Redfish handle authentication?"
    return "Apply BMC firmware 01.73.12 before the BIOS update [1]."


def _client(settings, llm: FakeLLM | None = None) -> TestClient:
    container = Container(
        settings=settings,
        store=MemoryVectorStore(),
        embedder=HashingEmbedder(),
        llm=llm or FakeLLM(_reply),
        reranker=None,
        token_counter=RegexTokenCounter(),
    )
    return TestClient(create_app(container))


def _upload(client: TestClient, name: str = "firmware-update.md", body: bytes | None = None):
    payload = body if body is not None else (
        b"# Firmware\n\nApply BMC firmware 01.73.12 before BIOS 2.8.4.\n"
    )
    return client.post("/documents", files={"file": (name, payload, "text/markdown")})


def test_view_rows_keep_scores() -> None:
    rows = hit_rows(
        [
            {
                "filename": "troubleshooting.md",
                "text": "Error 0x1F means SYS_FAN1 stalled",
                "section": "Errors",
                "page_start": 1,
                "semantic_score": 0.5,
                "bm25_score": 1.2,
                "fusion_score": 0.03,
                "rerank_score": None,
            }
        ]
    )
    assert rows[0]["filename"] == "troubleshooting.md"
    assert rows[0]["bm25_score"] == 1.2
    assert citation_rows([{"index": 1, "filename": "troubleshooting.md", "chunk_id": "c"}])[0]["id"] == 1


def test_health_ready_and_request_id(settings) -> None:
    client = _client(settings)
    health = client.get("/health", headers={"x-request-id": "req-123"})
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.headers["x-request-id"] == "req-123"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["vector_backend"] == "memory"
    assert ready.json()["reranker_enabled"] is False


def test_document_search_and_chat_round_trip(settings) -> None:
    client = _client(settings)
    created = _upload(client)
    assert created.status_code == 200
    document_id = created.json()["document_id"]
    assert created.json()["filename"] == "firmware-update.md"
    assert created.json()["chunks"] >= 1

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == document_id

    fetched = client.get(f"/documents/{document_id}")
    assert fetched.status_code == 200
    assert "01.73.12" in fetched.json()["chunks"][0]["text"]

    found = client.post("/search", json={"query": "01.73.12", "top_k": 3})
    assert found.status_code == 200
    assert found.json()["hits"]
    assert found.json()["hits"][0]["bm25_score"] is not None
    assert "retrieval_ms" in found.json()["timings_ms"]

    chat = client.post("/chat", json={"message": "Which BMC firmware comes first?"})
    body = chat.json()
    assert chat.status_code == 200
    assert body["rewritten_query"] == "Which BMC firmware comes first?"
    assert "01.73.12" in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["filename"] == "firmware-update.md"
    assert body["rejected_citation_ids"] == []
    assert "llm_ms" in body["timings_ms"]
    assert body["model_name"] == "fake-llm"

    follow = client.post(
        "/chat",
        json={"message": "What about authentication?", "conversation_id": body["conversation_id"]},
    )
    assert follow.status_code == 200
    assert follow.json()["rewritten_query"] == "How does Redfish handle authentication?"

    removed = client.delete(f"/documents/{document_id}")
    assert removed.status_code == 200
    missing = client.get(f"/documents/{document_id}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_upload_validation(settings) -> None:
    client = _client(settings)
    bad_type = client.post("/documents", files={"file": ("payload.exe", b"MZ", "application/octet-stream")})
    assert bad_type.status_code == 415
    empty = client.post("/documents", files={"file": ("empty.md", b"", "text/markdown")})
    assert empty.status_code == 422
    nested = _upload(client, name="../../etc/passwd.md", body=b"# Notes\n\nBMC hostname astrarack-bmc.local\n")
    assert nested.status_code == 200
    assert nested.json()["filename"] == "passwd.md"
    tiny = settings.model_copy(update={"max_upload_bytes": 8})
    limited = _client(tiny)
    large = limited.post("/documents", files={"file": ("big.md", b"0123456789abcdef", "text/markdown")})
    assert large.status_code == 413
