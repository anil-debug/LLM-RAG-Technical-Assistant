"""Chunking strategies.

Structure-aware chunking is the default for manuals: one section stays
together until it exceeds the token budget, and it never absorbs the next
heading. Fixed, token, and recursive strategies are available so tests and
the evaluation notes can compare them on the same text.
"""

import re
from dataclasses import dataclass

from core.text import token_spans
from ingestion.models import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_EVENT = re.compile(r"^\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Piece:
    """A span of text before it becomes a ``Chunk`` with an id."""

    text: str
    section: str | None
    page_start: int | None
    page_end: int | None


class TokenCounter:
    """Structural type: any object with ``count(text) -> int``."""

    def count(self, text: str) -> int:  # pragma: no cover - protocol placeholder
        raise NotImplementedError


def fixed_windows(text: str, size: int, overlap: int) -> list[str]:
    """Split on characters. ``overlap`` is a character count."""
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        raise ValueError("fixed chunk size must be positive.")
    overlap = min(max(overlap, 0), size - 1)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def token_windows(text: str, target: int, overlap: int) -> list[str]:
    """Split on tokenizer spans so a window does not cut a technical token."""
    stripped = text.strip()
    if not stripped:
        return []
    if overlap >= target:
        raise ValueError("overlap must be smaller than the target size.")
    spans = token_spans(stripped)
    if not spans:
        return [stripped]
    chunks: list[str] = []
    start = 0
    while start < len(spans):
        end = min(len(spans), start + target)
        piece = stripped[spans[start][0] : spans[end - 1][1]].strip()
        if piece:
            chunks.append(piece)
        if end == len(spans):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


def recursive_windows(text: str, counter: TokenCounter, target: int, overlap: int) -> list[str]:
    """Split on paragraph, line, sentence, then token windows."""
    text = text.strip()
    if not text:
        return []
    packed = _recursive(text, counter, target)
    return _prefix_overlap(packed, target, overlap)


def _recursive(text: str, counter: TokenCounter, target: int) -> list[str]:
    if counter.count(text) <= target:
        return [text]
    for separator in ("\n\n", "\n", ". ", " "):
        if separator not in text:
            continue
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        if len(parts) <= 1:
            continue
        packed: list[str] = []
        current = ""
        for part in parts:
            if counter.count(part) > target:
                if current:
                    packed.append(current)
                    current = ""
                packed.extend(_recursive(part, counter, target))
                continue
            candidate = part if not current else f"{current}{separator}{part}"
            if counter.count(candidate) <= target:
                current = candidate
            else:
                if current:
                    packed.append(current)
                current = part
        if current:
            packed.append(current)
        return packed
    return token_windows(text, target, overlap=0)


def _prefix_overlap(chunks: list[str], target: int, overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    output = [chunks[0]]
    for chunk in chunks[1:]:
        previous = output[-1]
        spans = token_spans(previous)
        if not spans:
            output.append(chunk)
            continue
        tail_start = spans[-overlap][0] if len(spans) > overlap else spans[0][0]
        tail = previous[tail_start:].strip()
        merged = f"{tail}\n{chunk}".strip()
        spans_merged = token_spans(merged)
        if len(spans_merged) <= target + overlap:
            output.append(merged)
        else:
            output.append(chunk)
    return output


def structure_pieces(
    document: Document,
    counter: TokenCounter,
    target: int,
    overlap: int,
) -> list[Piece]:
    """Chunk a manual by heading, or a PDF by page."""
    if document.media_type == "pdf" and document.pages:
        return _page_pieces(document, counter, target, overlap)
    return _heading_pieces(document.text, counter, target, overlap)


def log_pieces(text: str, counter: TokenCounter, target: int, overlap: int) -> list[Piece]:
    """Split a log on timestamped event boundaries."""
    events: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if _EVENT.match(line) and current:
            events.append(current)
            current = [line]
        elif line.strip():
            current.append(line)
    if current:
        events.append(current)
    pieces: list[Piece] = []
    for lines in events:
        event = "\n".join(lines).strip()
        section = lines[0][:48]
        pieces.extend(_pieces_for_block(event, section, None, None, counter, target, overlap))
    return pieces


def _page_pieces(document: Document, counter: TokenCounter, target: int, overlap: int) -> list[Piece]:
    pieces: list[Piece] = []
    for page in document.pages or []:
        if not page.text.strip():
            continue
        pieces.extend(
            _pieces_for_block(
                page.text,
                None,
                page.page_number,
                page.page_number,
                counter,
                target,
                overlap,
            )
        )
    return pieces


def _heading_pieces(text: str, counter: TokenCounter, target: int, overlap: int) -> list[Piece]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return _pieces_for_block(text, None, None, None, counter, target, overlap)
    pieces: list[Piece] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        pieces.extend(_pieces_for_block(preamble, None, None, None, counter, target, overlap))
    stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        section = " > ".join(item[1] for item in stack)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if not body:
            continue
        pieces.extend(_pieces_for_block(body, section, None, None, counter, target, overlap))
    return pieces


def _pieces_for_block(
    text: str,
    section: str | None,
    page_start: int | None,
    page_end: int | None,
    counter: TokenCounter,
    target: int,
    overlap: int,
) -> list[Piece]:
    text = text.strip()
    if not text:
        return []
    if counter.count(text) <= target:
        return [Piece(text, section, page_start, page_end)]
    return [
        Piece(part, section, page_start, page_end)
        for part in token_windows(text, target, overlap)
    ]
