"""Deterministic technical entity extraction.

Patterns and a small gazetteer are the right tool for versions, hex codes,
and fixed product terms. A learned tagger is a follow-on if labeled spans
show systematic misses. ``ARX-77`` is one such miss in the sample corpus:
it is a tracking code, and these patterns do not claim it.
"""

import re

from intelligence.schemas import Entity

# Earlier patterns are more specific and win when spans overlap.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("post_code", re.compile(r"\bPOST code\s+0x[0-9A-Fa-f]+\b", re.IGNORECASE)),
    ("product_firmware", re.compile(r"\bAR-[A-Z]+-\d+\.\d+\.\d+\b")),
    ("firmware_version", re.compile(r"\b\d+\.\d+\.\d+\b")),
    ("hex_code", re.compile(r"\b0x[0-9A-Fa-f]+\b")),
    ("error_code", re.compile(r"\b(?:NET|NVME|FAB)-\d+\b")),
    ("fru", re.compile(r"\bFRU(?:-[A-Z0-9]+)+\b")),
    ("identifier", re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")),
    ("endpoint", re.compile(r"/redfish/v1/[A-Za-z0-9_/{}-]*")),
)
_GAZETTEER = ("BMC", "BIOS", "IPMI", "Redfish", "NVMe", "NVIDIA", "firmware")


def extract_entities(text: str) -> list[Entity]:
    """Return non-overlapping entities. Gazetteer hits are unique per surface."""
    occupied: list[tuple[int, int]] = []
    found: list[Entity] = []
    for label, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if _overlaps(span, occupied):
                continue
            occupied.append(span)
            found.append(Entity(label, match.group(0), match.start(), match.end()))
    lowered = text.casefold()
    seen_terms: set[str] = set()
    for term in _GAZETTEER:
        needle = term.casefold()
        start = 0
        while True:
            index = lowered.find(needle, start)
            if index < 0:
                break
            end = index + len(needle)
            if needle not in seen_terms and not _overlaps((index, end), occupied):
                # Avoid matching inside a longer token.
                before = lowered[index - 1] if index else " "
                after = lowered[end] if end < len(lowered) else " "
                if not before.isalnum() and not after.isalnum():
                    found.append(Entity("gazetteer", text[index:end], index, end))
                    seen_terms.add(needle)
                    occupied.append((index, end))
            start = end
    found.sort(key=lambda entity: (entity.start, entity.label))
    return found


def unique_keys(entities: list[Entity] | list[dict]) -> set[tuple[str, str]]:
    """Compare entities by label and casefolded surface, ignoring offsets."""
    keys: set[tuple[str, str]] = set()
    for entity in entities:
        if isinstance(entity, Entity):
            keys.add((entity.label, entity.text.casefold()))
        else:
            keys.add((str(entity["label"]), str(entity["text"]).casefold()))
    return keys


def precision_recall(
    found: set[tuple[str, str]],
    gold: set[tuple[str, str]],
) -> dict[str, float | int | list]:
    """Set precision and recall. Empty gold with empty found scores 1.0."""
    true_positive = found & gold
    false_positive = found - gold
    false_negative = gold - found
    precision = len(true_positive) / len(found) if found else (1.0 if not gold else 0.0)
    recall = len(true_positive) / len(gold) if gold else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "true_positives": len(true_positive),
        "false_positives": sorted(false_positive),
        "false_negatives": sorted(false_negative),
    }


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in occupied)
