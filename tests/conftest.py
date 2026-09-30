"""Shared paths and a settings object that ignores the developer environment."""

from pathlib import Path

import pytest

from core.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        vector_backend="memory",
        reranker_enabled=False,
        intel_entities=False,
        intel_section_classifier=False,
        intel_document_classifier=False,
        intel_summarize=False,
        chunk_tokenizer="regex",
    )
