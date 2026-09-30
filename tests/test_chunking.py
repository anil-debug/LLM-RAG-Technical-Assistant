"""Chunk strategies keep different boundaries on the same manual."""

from core.text import RegexTokenCounter
from ingestion.chunking.chunker import chunk_document
from ingestion.cleaning import clean_document
from ingestion.loaders import load_file
from ingestion.metadata.extractor import extract_metadata
from ingestion.models import Document, Page
from tests.conftest import SAMPLE


def _manual():
    path = SAMPLE / "manuals" / "server-management.md"
    return extract_metadata(clean_document(load_file(path)))


def test_structure_does_not_merge_adjacent_sections():
    document = _manual()
    chunks = chunk_document(document, strategy="structure", counter=RegexTokenCounter(), target_tokens=480, overlap_tokens=72)
    factory = next(chunk for chunk in chunks if chunk.section and chunk.section.endswith("Factory reset"))
    assert "BIOS region" in factory.text
    assert "30 minutes" not in factory.text
    assert factory.token_count > 0
    assert chunks[0].next_chunk_id == chunks[1].id
    assert chunks[1].prev_chunk_id == chunks[0].id


def test_heading_path_includes_parents(tmp_path):
    path = tmp_path / "nested.md"
    path.write_text("# Root\n\n## Child\n\nThe child fact is 01.73.12.\n", encoding="utf-8")
    document = extract_metadata(clean_document(load_file(path)))
    chunks = chunk_document(document, strategy="structure", counter=RegexTokenCounter())
    child = next(chunk for chunk in chunks if "01.73.12" in chunk.text)
    assert child.section == "Root > Child"
    assert child.embed_text.startswith("Root > Child\n")


def test_token_windows_overlap_and_keep_version_tokens():
    words = " ".join(["alpha"] * 30 + ["01.73.12"] + ["beta"] * 30)
    document = Document(
        id="d",
        filename="n.txt",
        source="n.txt",
        media_type="text",
        checksum="abc",
        title="n",
        text=words,
        pages=None,
        created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    chunks = chunk_document(
        document,
        strategy="token",
        counter=RegexTokenCounter(),
        target_tokens=20,
        overlap_tokens=5,
    )
    assert len(chunks) > 1
    assert any("01.73.12" in chunk.text for chunk in chunks)
    # The version token is not split across a boundary inside itself.
    assert all("01.73.1" != chunk.text[-8:] for chunk in chunks)


def test_fixed_and_recursive_run():
    document = _manual()
    fixed = chunk_document(document, strategy="fixed", fixed_chars=180, fixed_overlap_chars=20)
    recursive = chunk_document(
        document,
        strategy="recursive",
        counter=RegexTokenCounter(),
        target_tokens=40,
        overlap_tokens=8,
    )
    assert fixed and recursive
    assert all(len(chunk.text) <= 180 for chunk in fixed)


def test_log_events_stay_intact():
    document = extract_metadata(clean_document(load_file(SAMPLE / "logs" / "sel.log")))
    chunks = chunk_document(document, strategy="structure", counter=RegexTokenCounter(), target_tokens=480, overlap_tokens=72)
    assert len(chunks) == 3
    assert any("0x1F" in chunk.text for chunk in chunks)
    assert any("NET-4401" in chunk.text for chunk in chunks)
    assert all("0x1F" not in chunk.text or "NET-4401" not in chunk.text for chunk in chunks)


def test_pdf_pages_become_page_citations():
    document = Document(
        id="pdf",
        filename="a.pdf",
        source="a.pdf",
        media_type="pdf",
        checksum="z",
        title="a",
        text="page one text\n\npage two text",
        pages=[Page(1, "page one text"), Page(2, "page two text")],
        created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    chunks = chunk_document(document, strategy="structure", counter=RegexTokenCounter())
    assert [chunk.page_start for chunk in chunks] == [1, 2]
