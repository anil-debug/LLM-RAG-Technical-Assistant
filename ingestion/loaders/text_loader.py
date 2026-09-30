"""Plain text and log loader."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ingestion.loaders.markdown_loader import _decode
from ingestion.models import Document


def load_text(path: Path, data: bytes, *, media_type: str = "text") -> Document:
    return Document(
        id=str(uuid4()),
        filename=path.name,
        source=str(path),
        media_type=media_type,
        checksum="",
        title="",
        text=_decode(data),
        pages=None,
        created_at=datetime.now(UTC),
    )
