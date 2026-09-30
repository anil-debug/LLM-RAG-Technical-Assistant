"""Tokenization shared by chunking and BM25.

Technical strings stay intact: ``0x1F``, ``01.73.12``, and ``SYS_FAN1`` are each
one token. Matching is case-insensitive because tokens are casefolded.
"""

import re

# Order matters. Hex and dotted versions must win over plain numbers.
TOKEN_RE = re.compile(
    r"0x[0-9a-fA-F]+"
    r"|\d+(?:\.\d+)+"
    r"|[A-Za-z][A-Za-z0-9_\-]*"
    r"|\d+"
)


def tokenize(text: str) -> list[str]:
    """Split ``text`` into casefolded technical tokens."""
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def token_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans for each token."""
    return [(match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


class RegexTokenCounter:
    """Count regex tokens. Used when the model tokenizer is not loaded."""

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(tokenize(text))
