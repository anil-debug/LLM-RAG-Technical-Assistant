"""Deterministic cleanup. This step does not call a model."""

import re
import unicodedata
from collections import Counter

from ingestion.models import Document, Page

_DEHYPHEN = re.compile(r"([A-Za-z])-\n([a-z])")


def clean_prose(text: str) -> str:
    """Normalize Unicode and whitespace, and join words split across lines.

    Fenced code blocks are left alone so log lines and commands keep their
    spacing. A hyphen is removed only when a line break sits between a letter
    and a lowercase letter, which is the usual PDF wrapping case.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _DEHYPHEN.sub(r"\1\2", text)
    parts = text.split("```")
    cleaned: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            cleaned.append(part)
            continue
        part = re.sub(r"[ \t]+\n", "\n", part)
        part = re.sub(r"[ \t]{2,}", " ", part)
        part = re.sub(r"\n{3,}", "\n\n", part)
        cleaned.append(part)
    return "```".join(cleaned).strip()


def drop_repeated_page_furniture(pages: list[Page]) -> list[Page]:
    """Remove short lines that repeat on most pages.

    Running headers and footers show up in nearly every page of a manual and
    drown both BM25 and embeddings. A line qualifies only when it is shorter
    than 80 characters and appears on at least 60% of pages. With fewer than
    three pages there is not enough evidence, so the pages are returned as they
    are.
    """
    if len(pages) < 3:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        unique = {
            line.strip()
            for line in page.text.splitlines()
            if 0 < len(line.strip()) < 80
        }
        counts.update(unique)
    threshold = len(pages) * 0.6
    repeated = {line for line, count in counts.items() if count >= threshold}
    cleaned: list[Page] = []
    for page in pages:
        kept = [line for line in page.text.splitlines() if line.strip() not in repeated]
        cleaned.append(Page(page.page_number, "\n".join(kept).strip()))
    return cleaned


def clean_document(document: Document) -> Document:
    """Clean page text and the flattened document text."""
    if document.pages:
        pages = [Page(page.page_number, clean_prose(page.text)) for page in document.pages]
        pages = drop_repeated_page_furniture(pages)
        document.pages = pages
        document.text = "\n\n".join(page.text for page in pages if page.text).strip()
    else:
        document.text = clean_prose(document.text)
    return document
