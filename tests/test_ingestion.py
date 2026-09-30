"""Loaders share one document shape and keep page or heading metadata."""

from pathlib import Path

from ingestion.loaders import load_file
from ingestion.metadata.extractor import extract_metadata
from ingestion.pipeline import prepare_document
from tests.conftest import SAMPLE


def test_markdown_txt_html_and_log_share_fields(settings):
    paths = [
        SAMPLE / "manuals" / "server-management.md",
        SAMPLE / "manuals" / "readme-notes.txt",
        SAMPLE / "manuals" / "account-service.html",
        SAMPLE / "logs" / "sel.log",
    ]
    media = []
    for path in paths:
        document, chunks = prepare_document(path, settings)
        assert document.id
        assert document.checksum
        assert document.filename == path.name
        assert document.created_at is not None
        assert chunks
        assert all(chunk.document_id == document.id for chunk in chunks)
        media.append(document.media_type)
    assert media == ["markdown", "text", "html", "log"]


def test_html_keeps_a_heading_section(settings):
    document, chunks = prepare_document(SAMPLE / "manuals" / "account-service.html", settings)
    assert document.title == "AstraRack account service notes"
    assert any(chunk.section and "Role boundaries" in chunk.section for chunk in chunks)
    assert any("Administrator" in chunk.text for chunk in chunks)


def test_pdf_keeps_page_numbers(tmp_path: Path, settings):
    pdf_path = tmp_path / "post.pdf"
    pdf_path.write_bytes(_tiny_pdf("POST code 0xD4 on CPU0_DIMM_A1"))
    document = extract_metadata(__import__("ingestion.cleaning", fromlist=["clean_document"]).clean_document(load_file(pdf_path)))
    assert document.media_type == "pdf"
    assert document.pages is not None
    assert document.pages[0].page_number == 1
    assert "0xD4" in document.pages[0].text
    prepared, chunks = prepare_document(pdf_path, settings)
    assert chunks[0].page_start == 1
    assert "0xD4" in prepared.text


def test_checksum_is_stable(settings):
    path = SAMPLE / "manuals" / "firmware-update.md"
    first, _ = prepare_document(path, settings)
    second, _ = prepare_document(path, settings)
    assert first.checksum == second.checksum
    assert first.id != second.id


def _tiny_pdf(message: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({message}) Tj ET".encode("latin-1")
    objects = [
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n",
        b"4 0 obj<</Length " + str(len(stream)).encode("ascii") + b">>stream\n" + stream + b"\nendstream\nendobj\n",
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
    ]
    header = b"%PDF-1.4\n"
    body = b""
    offsets = [0]
    cursor = len(header)
    for obj in objects:
        offsets.append(cursor)
        body += obj
        cursor += len(obj)
    xref_at = len(header) + len(body)
    xref = [b"xref\n0 6\n", b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n" + str(xref_at).encode("ascii") + b"\n%%EOF\n"
    return header + body + b"".join(xref) + trailer
