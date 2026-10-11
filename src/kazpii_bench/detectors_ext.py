"""Детекторы-сравнение: Presidio и Natasha — «из коробки», как их ставят клиенты."""
from __future__ import annotations

from .harness import Detected

# Presidio entity_type → типы бенча (ТОЛЬКО точные эквиваленты; US_* не маппятся —
# иначе KZ_IIN получал бы recall за счёт американских recognizers)
PRESIDIO_MAP = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CARD",
    "IBAN_CODE": "KZ_IBAN",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
    "URL": "URL",
    "IP_ADDRESS": "IP",
    "DATE_TIME": "DATE",
}


class PresidioDetector:
    """Microsoft Presidio из коробки: английские + встроенные recognizers, spaCy NER.

    Ожидаемый результат: почти ничего на казахских идентификаторах — это и есть
    измеряемый пробел рынка, который мы публикуем.
    """

    def __init__(self):
        from presidio_analyzer import AnalyzerEngine

        self.engine = AnalyzerEngine()
        self._langs = ("en",)

    def detect(self, text: str) -> list[Detected]:
        out = []
        for lang in self._langs:
            try:
                res = self.engine.analyze(text=text, language=lang)
            except Exception:
                continue
            for r in res:
                out.append(Detected(PRESIDIO_MAP.get(r.entity_type, r.entity_type),
                                    text[r.start:r.end], r.start, r.end))
        return out


class NatashaDetector:
    """Natasha NER (русский): PER/ORG/LOC. Даёт офсеты."""

    def __init__(self):
        from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter

        self._seg = Segmenter()
        self._emb = NewsEmbedding()
        self._ner = NewsNERTagger(self._emb)
        self._doc_cls = Doc

    def detect(self, text: str) -> list[Detected]:
        doc = self._doc_cls(text)
        doc.segment(self._seg)
        doc.tag_ner(self._ner)
        out = []
        for span in doc.spans:
            t = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}.get(span.type)
            if t:
                out.append(Detected(t, text[span.start:span.stop], span.start, span.stop))
        return out


class GlinerDetector:
    """GLiNER zero-shot (urchade/gliner_multi_pii-v1, Apache-2.0, без ru/kk).

    Тяжёлые зависимости ставятся отдельно: pip install gliner torch.
    """

    LABELS = ["person", "organization", "iin", "bin", "iban", "credit card", "phone", "email"]

    def __init__(self, model: str = "urchade/gliner_small-v2.1", threshold: float = 0.4):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model)
        self.threshold = threshold

    def detect(self, text: str) -> list[Detected]:
        raw = self.model.predict_entities(text, self.LABELS, threshold=self.threshold)
        label_map = {
            "person": "PERSON", "organization": "ORG", "iin": "KZ_IIN", "bin": "KZ_BIN",
            "iban": "KZ_IBAN", "credit card": "CARD", "phone": "PHONE", "email": "EMAIL",
        }
        return [Detected(label_map.get(r["label"], r["label"].upper()), r["text"], r["start"], r["end"])
                for r in raw]
