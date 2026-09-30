"""Cleaning is deterministic and keeps technical tokens intact."""

from ingestion.cleaning import clean_prose, drop_repeated_page_furniture
from ingestion.models import Page


def test_dehyphenates_wrapped_words_and_normalizes_unicode():
    text = "The firm-\nware image is 01.73.12.\n\n\nNext."
    cleaned = clean_prose(text)
    assert "firmware" in cleaned
    assert "01.73.12" in cleaned
    assert "\n\n\n" not in cleaned


def test_code_fence_whitespace_is_preserved():
    text = "Intro\n```\nSYS_FAN1   stalled\n```\nAfter"
    cleaned = clean_prose(text)
    assert "SYS_FAN1   stalled" in cleaned


def test_repeated_headers_drop_only_with_enough_pages():
    pages = [
        Page(1, "AstraRack manual\nBMC stays powered"),
        Page(2, "AstraRack manual\nVLAN 40"),
        Page(3, "AstraRack manual\nError 0x1F"),
    ]
    cleaned = drop_repeated_page_furniture(pages)
    assert all("AstraRack manual" not in page.text for page in cleaned)
    assert "0x1F" in cleaned[2].text
