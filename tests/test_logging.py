"""Log lines include level, logger name, message, and the current request id."""

import logging

from core.context import bind_request
from core.logging_config import log_event, setup_logging


def test_request_id_appears_in_log_line(capsys):
    setup_logging("INFO")
    with bind_request("req-123"):
        logging.getLogger("tests.sample").info("hello-pipeline")
    captured = capsys.readouterr().err
    assert "req-123" in captured
    assert "INFO" in captured
    assert "tests.sample" in captured
    assert "hello-pipeline" in captured
    assert "request_id=req-123" in captured


def test_missing_ids_are_dashes(capsys):
    setup_logging("INFO")
    logging.getLogger("tests.sample").info("no-context")
    captured = capsys.readouterr().err
    assert "request_id=-" in captured
    assert "query_id=-" in captured
    assert "document_id=-" in captured


def test_log_event_includes_fields(capsys):
    setup_logging("INFO")
    with bind_request("req-9"):
        log_event(
            logging.getLogger("retrieval.hybrid"),
            "search complete",
            retrieval_ms=12.5,
            model_name="BAAI/bge-base-en-v1.5",
            top_k=8,
        )
    captured = capsys.readouterr().err
    assert "retrieval_ms=12.5" in captured
    assert "top_k=8" in captured
    assert "request_id=req-9" in captured
