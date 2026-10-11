"""Детекторы для харнеса."""
from __future__ import annotations

import re

from .harness import Detected


class BaselineDetector:
    """Regex + контрольные суммы через kz-identifiers — «планка», которую любая
    серьёзная система обязана бить. NER-типы (PERSON/ORG/LOC) не детектируются.
    """

    def __init__(self):
        from kz_identifiers import (
            is_valid_iin_checksum,
            luhn_ok,
            parse_iban,
            validate_bin,
        )
        self._ok_iin = is_valid_iin_checksum
        self._bin = validate_bin
        self._iban = parse_iban
        self._luhn = luhn_ok
        self._re_iban = re.compile(r"\bKZ(?:\s?\d){2}(?:\s?\d){3}(?:\s?[A-Z0-9]){13}\b")
        self._re_card = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
        self._re_phone = re.compile(r"(?<!\d)(?:\+7|8)[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-()]?\d{2}[\s\-()]?\d{2}(?!\d)")
        self._re_email = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
        self._re_pp = re.compile(r"\b(?:[A-Z]\d{7}|[A-Z]\d{2}\s\d{6})\b")
        self._re_12 = re.compile(r"(?<!\d)\d{12}(?!\d)")

    def detect(self, text: str) -> list[Detected]:
        out = []
        for m in self._re_iban.finditer(text):
            if self._iban(m.group(0)).valid:
                out.append(Detected("KZ_IBAN", m.group(0), m.start(), m.end()))
        for m in self._re_12.finditer(text):
            v = m.group(0)
            if self._bin(v, strict_facets=True):
                out.append(Detected("KZ_BIN", v, m.start(), m.end()))
            elif self._ok_iin(v):
                out.append(Detected("KZ_IIN", v, m.start(), m.end()))
        for m in self._re_card.finditer(text):
            if self._luhn(m.group(0)):
                out.append(Detected("CARD", m.group(0), m.start(), m.end()))
        for m in self._re_phone.finditer(text):
            out.append(Detected("PHONE", m.group(0), m.start(), m.end()))
        for m in self._re_email.finditer(text):
            out.append(Detected("EMAIL", m.group(0), m.start(), m.end()))
        for m in self._re_pp.finditer(text):
            out.append(Detected("KZ_PASSPORT", m.group(0), m.start(), m.end()))
        return out


class GatewayScanDetector:
    """HTTP-детектор через /scan шлюза (read-only, без записи в vault).

    /scan возвращает [{"type", "score", "sample_masked"}] без офсетов, где
    sample_masked = первые2 + ** + последние2 + (sha1). Матчим по
    (type, first2, last2) — этого достаточно для value-режима.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        import json as _json
        import urllib.request

        self._json = _json
        self._req = urllib.request.Request
        self._urlopen = urllib.request.urlopen
        self.url = base_url.rstrip("/")
        self.timeout = timeout

    def detect(self, text: str) -> list[Detected]:
        req = self._req(self.url + "/scan", data=self._json.dumps({"texts": [text]}).encode(),
                        headers={"Content-Type": "application/json"})
        with self._urlopen(req, timeout=self.timeout) as resp:
            res = self._json.loads(resp.read().decode())
        out = []
        for f in res.get("findings", []):
            sample = f.get("sample_masked", "")
            # "KZ**00(5026a2)" -> first2="KZ", last2="00"
            m = re.match(r"^(.{0,2})\*\*(.{0,2})\(", sample)
            first2, last2 = (m.group(1), m.group(2)) if m else ("", "")
            out.append(Detected(f["type"], f"{first2}|{last2}".upper()))
        return out

