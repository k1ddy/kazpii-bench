"""KazPII-Bench — open benchmark for PII detection in Kazakh and Russian texts.

Layer A generates synthetic identifiers inside real-looking contexts.
Schema (JSONL, one record per line):

    {
      "id": "kzpb-A000001",
      "text": "…",
      "spans": [{"start": 12, "end": 24, "type": "KZ_IIN", "value": "900517301146"}],
      "layer": "A",
      "synthetic": true,
      "generator": "kz-identifiers 0.1.0",
      "context": "whatsapp",
      "noise": ["ocr"],
      "canary": "…",
      "split": "dev"
    }

Invariants (tested):
  - spans are exact: text[start:end] == value
  - all noise is 1:1 character substitution (offsets never shift)
  - generation is deterministic given (seed, version)
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

CANARY = "kzpii-bench-canary-2026"
SCHEMA_VERSION = "0.1.0"


@dataclass
class Span:
    start: int
    end: int
    type: str
    value: str


@dataclass
class Example:
    id: str
    text: str
    spans: list[Span] = field(default_factory=list)
    layer: str = "A"
    synthetic: bool = True
    generator: str = "kz-identifiers 0.1.0"
    context: str = ""
    noise: list[str] = field(default_factory=list)
    lang: str = "ru"
    split: str = "dev"
    canary: str = CANARY
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        for s in self.spans:
            assert self.text[s.start:s.end] == s.value, (
                f"span mismatch: {self.text[s.start:s.end]!r} != {s.value!r}"
            )
            assert s.start < s.end, "empty span"

    def to_jsonl(self) -> str:
        self.validate()
        d = asdict(self)
        d["spans"] = [asdict(s) for s in self.spans]
        return json.dumps(d, ensure_ascii=False)


def write_jsonl(examples: Iterator[Example], path) -> int:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_jsonl() + "\n")
            n += 1
    return n


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
