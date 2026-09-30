"""Fill title and descriptive metadata from text we already extracted."""

import re
from pathlib import Path

from ingestion.models import Document

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def extract_metadata(document: Document) -> Document:
    """Set the title and copy stable facts into ``metadata``.

    The title is the first Markdown heading. PDF and plain text fall back to
    the filename stem. Page count is recorded when the loader kept pages.
    """
    match = _HEADING.search(document.text)
    document.title = match.group(1).strip() if match else Path(document.filename).stem
    document.metadata["page_count"] = len(document.pages) if document.pages else None
    document.metadata["media_type"] = document.media_type
    document.metadata["checksum"] = document.checksum
    document.metadata.setdefault("intelligence", {})
    return document
