"""Annotations produced by the optional document-intelligence modules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    label: str
    text: str
    start: int
    end: int

    def as_dict(self) -> dict[str, str | int]:
        return {"label": self.label, "text": self.text, "start": self.start, "end": self.end}
