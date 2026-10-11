"""KazPII-Bench — open benchmark for PII detection in Kazakh and Russian texts."""

from .schema import CANARY, SCHEMA_VERSION, Example, Span, write_jsonl

__version__ = "0.1.0"
__all__ = ["CANARY", "SCHEMA_VERSION", "Example", "Span", "__version__", "write_jsonl"]
