"""Document intelligence: patterns, a linear head, and independent flags."""

from pathlib import Path

import numpy as np
import pytest

from core.errors import ClassifierNotReadyError, ModelUnavailableError
from core.settings import Settings
from ingestion.pipeline import prepare_document
from intelligence.entities import extract_entities, precision_recall, unique_keys
from intelligence.linear_classifier import train_classifier
from intelligence.section_classifier import SECTION_LABELS
from tests.conftest import SAMPLE
from tests.fakes import FakeLLM, HashingEmbedder


def test_patterns_keep_technical_spans_and_miss_arx_77() -> None:
    text = (
        "Error 0x1F means SYS_FAN1 stalled. Replace FRU FRU-FAN-220. "
        "POST code 0xD4 failed on CPU0_DIMM_A1. "
        "Bundle AR-GPU-1.6.0 and BMC firmware 01.73.12. "
        "Tracking code ARX-77 is not a version."
    )
    found = extract_entities(text)
    keys = unique_keys(found)
    assert ("hex_code", "0x1f") in keys
    assert ("identifier", "sys_fan1") in keys
    assert ("fru", "fru-fan-220") in keys
    assert ("post_code", "post code 0xd4") in keys
    assert ("product_firmware", "ar-gpu-1.6.0") in keys
    assert ("firmware_version", "01.73.12") in keys
    assert ("gazetteer", "bmc") in keys
    assert all(entity.text.casefold() != "arx-77" for entity in found)
    root = extract_entities("The service root is /redfish/v1/. Clients send JSON.")
    assert any(entity.text == "/redfish/v1/" for entity in root)
    assert all(not entity.text.endswith(".") for entity in root)
    # The post-code pattern owns the hex span, so 0xD4 is not a second entity.
    assert ("hex_code", "0xd4") not in keys


def test_entity_precision_recall_counts_the_tracking_code_miss() -> None:
    text = "This guide covers release train ARX-77. ARX-77 is not a firmware version."
    found = unique_keys(extract_entities(text))
    gold = {("gazetteer", "firmware"), ("tracking_code", "arx-77")}
    scores = precision_recall(found, gold)
    assert ("tracking_code", "arx-77") in scores["false_negatives"]
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(0.5)


def test_linear_head_separates_frozen_features() -> None:
    rng = np.random.default_rng(0)
    overview = rng.normal(loc=0.0, scale=0.05, size=(24, 8)).astype(np.float32)
    procedure = rng.normal(loc=3.0, scale=0.05, size=(24, 8)).astype(np.float32)
    features = np.vstack([overview, procedure])
    targets = np.array([0] * 24 + [1] * 24)
    trained = train_classifier(features, targets, ["overview", "procedure"], epochs=60)
    predictions = trained.predict(features)
    assert predictions[:24] == ["overview"] * 24
    assert predictions[24:] == ["procedure"] * 24
    assert "error_definition" in SECTION_LABELS


def test_entity_flag_annotates_without_a_classifier(settings) -> None:
    enabled = settings.model_copy(update={"intel_entities": True})
    document, chunks = prepare_document(SAMPLE / "manuals" / "firmware-update.md", enabled)
    surfaces = {entity["text"] for entity in document.metadata["intelligence"]["entities"]}
    assert "01.73.12" in surfaces
    assert "ARX-77" not in surfaces
    assert chunks[0].metadata["entities"]


def test_section_classifier_refuses_missing_weights(settings, tmp_path: Path) -> None:
    enabled = settings.model_copy(
        update={
            "intel_section_classifier": True,
            "section_classifier_path": tmp_path / "missing.pt",
        }
    )
    with pytest.raises(ClassifierNotReadyError):
        prepare_document(
            SAMPLE / "manuals" / "readme-notes.txt",
            enabled,
            embedder=HashingEmbedder(),
        )


def test_summary_chunk_requires_an_llm_and_can_be_added(settings) -> None:
    enabled = settings.model_copy(update={"intel_summarize": True})
    with pytest.raises(ModelUnavailableError):
        prepare_document(SAMPLE / "manuals" / "readme-notes.txt", enabled, llm=None)
    document, chunks = prepare_document(
        SAMPLE / "manuals" / "readme-notes.txt",
        enabled,
        llm=FakeLLM("The hostname astrarack-bmc.local is the DNS fallback."),
    )
    assert document.metadata["intelligence"]["summary"].startswith("The hostname")
    assert chunks[-1].section == "Summary"
    assert chunks[-1].metadata["kind"] == "summary"


def test_flags_default_off() -> None:
    settings = Settings(_env_file=None)
    assert settings.intel_entities is False
    assert settings.intel_section_classifier is False
    assert settings.intel_document_classifier is False
    assert settings.intel_summarize is False
