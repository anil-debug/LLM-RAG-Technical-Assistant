"""Standard-library logging with request, query, and document ids.

Records look like::

    2026-09-30T12:00:00+0000 INFO retrieval.hybrid request_id=abc query_id=def document_id=- fused hits=8
"""

import logging

from core.context import get_document_id, get_query_id, get_request_id

LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "request_id=%(request_id)s query_id=%(query_id)s document_id=%(document_id)s "
    "%(message)s"
)


class RequestIdFilter(logging.Filter):
    """Copy context-variable ids onto every log record.

    Formatters can then print ``request_id`` even when the caller did not
    pass ``extra``. A missing id is printed as ``-``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", None) or get_request_id() or "-"
        record.query_id = getattr(record, "query_id", None) or get_query_id() or "-"
        record.document_id = getattr(record, "document_id", None) or get_document_id() or "-"
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once for process-wide structured lines."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%z"))
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(level.upper())


def log_event(logger: logging.Logger, message: str, **fields: object) -> None:
    """Log ``message`` followed by ``key=value`` fields.

    Use this for stage timings and ids that an operator needs to grep.
    """
    rendered = " ".join(f"{key}={value}" for key, value in fields.items())
    if rendered:
        logger.info("%s %s", message, rendered)
    else:
        logger.info("%s", message)
