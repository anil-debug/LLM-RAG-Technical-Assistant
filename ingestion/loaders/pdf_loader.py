"""PDF loader. Page numbers are kept for citations."""

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from ingestion.models import Document, Page


def load_pdf(path: Path, data: bytes) -> Document:
    """Extract text per page. The checksum is filled in by ``load_file``."""
    reader = PdfReader(BytesIO(data))
    pages = [
        Page(page_number=index, text=page.extract_text() or "")
        for index, page in enumerate(reader.pages, start=1)
    ]
    text = "\n\n".join(page.text for page in pages if page.text)
    return Document(
        id=str(uuid4()),
        filename=path.name,
        source=str(path),
        media_type="pdf",
        checksum="",
        title="",
        text=text,
        pages=pages,
        created_at=datetime.now(UTC),
    )
