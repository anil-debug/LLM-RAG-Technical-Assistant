"""Dispatch a file to the loader for its suffix."""

import hashlib
from pathlib import Path

from core.errors import EmptyDocumentError, UnsupportedMediaTypeError
from ingestion.loaders.html_loader import load_html
from ingestion.loaders.markdown_loader import load_markdown
from ingestion.loaders.pdf_loader import load_pdf
from ingestion.loaders.text_loader import load_text
from ingestion.models import Document

ALLOWED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm", ".log"}


def load_file(path: Path) -> Document:
    """Load ``path`` into a ``Document`` and set the SHA-256 checksum."""
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedMediaTypeError(f"Unsupported file type '{suffix}'.")
    data = path.read_bytes()
    if not data:
        raise EmptyDocumentError(f"{path.name} is empty.")
    if suffix == ".pdf":
        document = load_pdf(path, data)
    elif suffix in {".md", ".markdown"}:
        document = load_markdown(path, data)
    elif suffix in {".html", ".htm"}:
        document = load_html(path, data)
    elif suffix == ".log":
        document = load_text(path, data, media_type="log")
    else:
        document = load_text(path, data, media_type="text")
    document.checksum = hashlib.sha256(data).hexdigest()
    return document
