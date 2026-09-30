"""Query rewriting uses history only to build a standalone retrieval string."""

from generation.rewrite import rewrite_query
from tests.fakes import FakeLLM


def test_no_history_skips_the_model() -> None:
    llm = FakeLLM("should not be called")
    assert rewrite_query("  How is authentication handled?  ", [], llm) == "How is authentication handled?"
    assert llm.calls == []


def test_follow_up_becomes_the_model_first_line() -> None:
    llm = FakeLLM("How does Redfish handle authentication?\nextra commentary")
    history = [
        {"role": "user", "content": "What is Redfish?"},
        {"role": "assistant", "content": "Redfish is the HTTPS JSON API."},
    ]
    rewritten = rewrite_query("What about authentication?", history, llm)
    assert rewritten == "How does Redfish handle authentication?"
    assert "Evidence" not in llm.calls[0][-1]["content"]
    assert llm.calls[0][0]["role"] == "system"


def test_empty_or_huge_rewrite_falls_back_to_the_question() -> None:
    history = [{"role": "user", "content": "Earlier turn"}]
    assert rewrite_query("What about Redfish?", history, FakeLLM("   ")) == "What about Redfish?"
    assert rewrite_query("What about Redfish?", history, FakeLLM("x" * 401)) == "What about Redfish?"
    assert rewrite_query("What about Redfish?", history, None) == "What about Redfish?"
