"""Markdown loader. Headings stay in the text for structure-aware chunking."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from core.errors import UnsupportedMediaTypeError
from ingestion.models import Document


def load_markdown(path: Path, data: bytes) -> Document:
    text = _decode(data)
    return Document(
        id=str(uuid4()),
        filename=path.name,
        source=str(path),
        media_type="markdown",
        checksum="",
        title="",
        text=text,
        pages=None,
        created_at=datetime.now(UTC),
    )


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnsupportedMediaTypeError("Text files must be UTF-8.") from exc
