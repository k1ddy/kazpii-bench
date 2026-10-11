"""Слой B: KazNERD — реальные казахские тексты + вставленный синтетический идентификатор.

Источник: KazNERD (ISSAI Nazarbayev University), 112 702 предложения, CC BY 4.0.
Метод: реальный контекст + синтетический идентификатор из kz-identifiers.
Предложения без вставки идут как негативные примеры: в них есть реальные
PER/ORG/LOC, но нет персональных идентификаторов — это ловит ложные
срабатывания детекторов на именах и организациях.
"""
from __future__ import annotations

import random

from kz_identifiers import (
    generate_bin,
    generate_card,
    generate_iin,
    random_iban,
)

from .layer_a import finalize_spans, gen_email, gen_passport_kz, gen_phone
from .schema import Example, Span

KAZNERD_SOURCE = "KazNERD (ISSAI Nazarbayev University), CC BY 4.0"
MAX_LEN = 512

# маркеры вставки, нейтральные для казахского текста
INSERT_MARKERS_KZ = ["ИИН ", "ЖСН ", "счет ", "шот "]
INSERT_MARKERS_NUM = ["№ ", "# "]


def iter_sentences(path):
    """CoNLL-2002 (IOB2): token\\ttag, предложения разделены пустой строкой."""
    tokens, tags, sent_id = [], [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    yield sent_id, tokens, tags
                    sent_id += 1
                    tokens, tags = [], []
                continue
            parts = line.split()
            token = parts[0]
            tag = parts[1] if len(parts) > 1 else "O"
            tokens.append(token)
            tags.append(tag)
    if tokens:
        yield sent_id, tokens, tags


def build_example(rng, sent_id, tokens, idx, *, insert_prob=0.7) -> Example | None:
    text = " ".join(tokens)
    if not 20 <= len(text) <= MAX_LEN:
        return None

    spans: list[Span] = []
    if rng.random() < insert_prob:
        kind = rng.choice(["iin", "phone", "iban", "card", "bin", "email", "passport"])
        if kind == "iin":
            value, marker = generate_iin(rng=rng), rng.choice(INSERT_MARKERS_KZ[:2])
        elif kind == "phone":
            value, marker = gen_phone(rng), ""
        elif kind == "iban":
            value, marker = random_iban(rng=rng), rng.choice(INSERT_MARKERS_KZ[2:])
        elif kind == "card":
            value, marker = generate_card(rng=rng), rng.choice(INSERT_MARKERS_NUM)
        elif kind == "bin":
            value, marker = generate_bin(rng=rng), rng.choice(INSERT_MARKERS_NUM)
        elif kind == "email":
            value, marker = gen_email(rng), ""
        else:
            value, marker = gen_passport_kz(rng), "паспорт "

        insertion = f"{marker}{value}"
        if rng.random() < 0.5:  # вставка в конец предложения
            start = len(text)
            text = f"{text} {insertion}"
        else:  # вставка после случайного слова в первой половине
            pos = rng.randrange(1, max(2, len(tokens) // 2))
            head = " ".join(tokens[:pos])
            tail = " ".join(tokens[pos:])
            text = f"{head} {insertion} {tail}"
            start = len(head) + 1
        spans.append(Span(start=start, end=start + len(insertion), type=f"PLACEHOLDER:{kind}", value=insertion))
        spans = finalize_spans(spans)
        for s in spans:
            s.value = text[s.start:s.end]

    return Example(
        id=f"kzpb-B{idx:06d}",
        text=text,
        spans=spans,
        layer="B",
        synthetic=bool(spans),
        generator="kz-identifiers/0.1.0" if spans else "",
        context="news_kk",
        lang="kk",
        split="dev",
    )


def ingest(path: str, n: int, seed: int, test_ratio: float = 0.2):
    from .layer_a import split_for

    rng = random.Random(seed)
    dev, test, seen = [], [], 0
    for sent_id, tokens, tags in iter_sentences(path):
        if seen >= n:
            break
        ex = build_example(rng, sent_id, tokens, seen)
        seen += 1
        if ex is None:
            continue
        split = split_for(seed, seen, test_ratio)
        ex.split = split
        (test if split == "test" else dev).append(ex)
    return dev, test
