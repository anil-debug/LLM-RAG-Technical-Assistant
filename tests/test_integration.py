"""In-process RAG loop over the synthetic corpus. No external model server."""

from core.text import RegexTokenCounter
from evaluation.corpus import index_documents
from generation.answer import answer_question
from retrieval.hybrid_search import hybrid_retrieve
from tests.fakes import FakeLLM, HashingEmbedder


def test_sample_corpus_answers_an_exact_identifier(settings) -> None:
    embedder = HashingEmbedder()
    store = index_documents(settings, embedder=embedder)
    hits = hybrid_retrieve(
        "What does error 0x1F mean?",
        embedder=embedder,
        store=store,
        candidates=30,
        top_n=8,
        rrf_k=60,
    )
    assert any("SYS_FAN1" in hit.text and hit.filename == "troubleshooting.md" for hit in hits)

    def reply(messages: list[dict[str, str]]) -> str:
        evidence = messages[-1]["content"]
        assert "SYS_FAN1" in evidence
        assert "Ignore previous instructions" not in messages[0]["content"]
        return "Error 0x1F means SYS_FAN1 stalled [1]."

    result = answer_question(
        "What does error 0x1F mean?",
        history=[],
        embedder=embedder,
        store=store,
        llm=FakeLLM(reply),
        counter=RegexTokenCounter(),
        settings=settings,
        conversation_id=None,
    )
    assert result.citations
    assert result.rejected_citation_ids == []
    assert "SYS_FAN1" in result.answer
    assert result.citations[0].filename == "troubleshooting.md"
    assert "retrieval_ms" in result.timings_ms


def test_follow_up_retrieval_uses_the_rewritten_query(settings) -> None:
    embedder = HashingEmbedder()
    store = index_documents(settings, embedder=embedder)
    seen: list[str] = []

    def reply(messages: list[dict[str, str]]) -> str:
        last = messages[-1]["content"]
        if "standalone search query" in last:
            return "Which FRU replaces SYS_FAN1 after error 0x1F?"
        seen.append(last)
        return "Replace FRU-FAN-220 [1]."

    history = [
        {"role": "user", "content": "Tell me about error 0x1F."},
        {"role": "assistant", "content": "Error 0x1F means SYS_FAN1 stalled."},
    ]
    result = answer_question(
        "What part do I replace?",
        history=history,
        embedder=embedder,
        store=store,
        llm=FakeLLM(reply),
        counter=RegexTokenCounter(),
        settings=settings,
    )
    assert result.rewritten_query == "Which FRU replaces SYS_FAN1 after error 0x1F?"
    assert "FRU-FAN-220" in result.answer
    assert seen
    assert "What part do I replace?" in seen[0]
    assert "Tell me about error 0x1F." not in seen[0].split("Evidence", 1)[-1]
