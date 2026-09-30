"""Citation ids are resolved only against the packed context."""

from generation.answer import extract_citation_ids, render_answer, resolve_citations
from generation.context import ContextBlock, ContextPack, build_context
from core.text import RegexTokenCounter
from retrieval.types import Hit, StoredChunk


def _hit(chunk_id: str, text: str, filename: str = "firmware-update.md") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        document_id="doc",
        filename=filename,
        title="Firmware",
        text=text,
        section="BMC and BIOS order",
        page_start=2,
        page_end=2,
    )


def test_invalid_citation_ids_are_rejected() -> None:
    hit = _hit("c1", "Apply BMC firmware 01.73.12 before BIOS 2.8.4.")
    pack = ContextPack(blocks=[ContextBlock(index=1, hit=hit, text=hit.text)])
    valid, rejected = resolve_citations("Use 01.73.12 [1]. Also see [9] and [1].", pack)
    assert extract_citation_ids("Use 01.73.12 [1]. Also see [9] and [1].") == [1, 9]
    assert [item.index for item in valid] == [1]
    assert rejected == [9]
    assert valid[0].filename == "firmware-update.md"
    assert valid[0].section == "BMC and BIOS order"
    assert valid[0].page_start == 2
    assert valid[0].chunk_id == "c1"


def test_render_drops_rejected_markers_and_appends_sources() -> None:
    hit = _hit("c1", "Apply BMC firmware 01.73.12.")
    citation, rejected = resolve_citations("Answer [1] and [4].", ContextPack(blocks=[ContextBlock(1, hit, hit.text)]))
    rendered = render_answer("Answer [1] and [4].\n\nSources:\nmodel invented this", citation, rejected)
    assert "[4]" not in rendered
    assert "[1]" in rendered
    assert "model invented this" not in rendered
    assert "chunk c1" in rendered
    assert "firmware-update.md" in rendered


def test_context_budget_keeps_the_first_block_and_numbers_from_one() -> None:
    hits = [
        _hit("c1", "alpha " * 30),
        _hit("c2", "beta " * 30),
    ]
    pack = build_context(hits, counter=RegexTokenCounter(), budget=5)
    assert [block.index for block in pack.blocks] == [1]
    assert pack.prompt_text.startswith("[1]")


def test_context_appends_the_following_chunk_in_the_same_section() -> None:
    first = _hit("c1", "Apply BMC firmware 01.73.12.")
    stored = {
        "c1": StoredChunk(
            chunk_id="c1",
            document_id="doc",
            filename="firmware-update.md",
            title="Firmware",
            text=first.text,
            section="BMC and BIOS order",
            page_start=2,
            page_end=2,
            embed_text=first.text,
            token_count=4,
            prev_chunk_id=None,
            next_chunk_id="c2",
        ),
        "c2": StoredChunk(
            chunk_id="c2",
            document_id="doc",
            filename="firmware-update.md",
            title="Firmware",
            text="BIOS 2.8.4 follows.",
            section="BMC and BIOS order",
            page_start=2,
            page_end=2,
            embed_text="BIOS 2.8.4 follows.",
            token_count=3,
            prev_chunk_id="c1",
            next_chunk_id=None,
        ),
    }
    pack = build_context([first], counter=RegexTokenCounter(), budget=200, chunks_by_id=stored)
    assert "2.8.4" in pack.blocks[0].text
    assert "2.8.4" in pack.prompt_text
