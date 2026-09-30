"""HTML loader.

Scripts, styles, and navigation are dropped. Headings become Markdown headings
so the structure-aware chunker can keep the section path.
"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from bs4 import BeautifulSoup

from ingestion.loaders.markdown_loader import _decode
from ingestion.models import Document


def load_html(path: Path, data: bytes) -> Document:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()
    for level in range(1, 4):
        for heading in soup.find_all(f"h{level}"):
            label = heading.get_text(" ", strip=True)
            heading.replace_with(f"\n{'#' * level} {label}\n")
    text = _decode(soup.get_text("\n").encode("utf-8"))
    return Document(
        id=str(uuid4()),
        filename=path.name,
        source=str(path),
        media_type="html",
        checksum="",
        title="",
        text=text,
        pages=None,
        created_at=datetime.now(UTC),
    )
