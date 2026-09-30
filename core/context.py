"""Context variables that logging and request handlers share.

Application code calls ``set_request_id`` at the start of a request and
``reset_request_id`` when the request finishes. Log records read the current
value through ``RequestIdFilter`` in ``core.logging_config``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from uuid import uuid4

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_query_id: ContextVar[str | None] = ContextVar("query_id", default=None)
_document_id: ContextVar[str | None] = ContextVar("document_id", default=None)


def get_request_id() -> str | None:
    return _request_id.get()


def get_query_id() -> str | None:
    return _query_id.get()


def get_document_id() -> str | None:
    return _document_id.get()


def set_request_id(value: str | None) -> Token[str | None]:
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


def set_query_id(value: str | None) -> Token[str | None]:
    return _query_id.set(value)


def reset_query_id(token: Token[str | None]) -> None:
    _query_id.reset(token)


def set_document_id(value: str | None) -> Token[str | None]:
    return _document_id.set(value)


def reset_document_id(token: Token[str | None]) -> None:
    _document_id.reset(token)


@contextmanager
def bind_request(request_id: str | None = None) -> Iterator[str]:
    """Attach a request id for the duration of the ``with`` block."""
    current = request_id or uuid4().hex
    token = set_request_id(current)
    try:
        yield current
    finally:
        reset_request_id(token)
