"""Слой A: программная генерация синтетических примеров с точными офсетами.

Инварианты (проверяются в тестах):
  - каждая вставка значения пишется через отслеживание длины буфера → офсеты точны;
  - весь нойз — только 1:1 подстановки, офсеты не сдвигаются;
  - генерация детерминирована при фиксированном seed;
  - в тестовую выборку не попадает ни один dev-пример (split по хешу).
"""
from __future__ import annotations

import hashlib
import random

from kz_identifiers import (
    generate_bin,
    generate_card,
    generate_iin,
    random_iban,
    validate_iin,
)

from .contexts import BODIES, CONTEXTS, LANG_BY_CONTEXT
from .noise import apply_noise_to_span
from .schema import Example, Span

FIRST_KZ = ["Аслан", "Дархан", "Ерлан", "Жанат", "Айгерім", "Мейір", "Нұрсұлтан", "Алия"]
LAST_KZ = ["Сапарбаев", "Оспанов", "Қасымов", "Тлеубаев", "Ахметов", "Смағұлов"]
KZ_SUFFIX = ["ұлы", "қызы"]
FIRST_RU = ["Иван", "Мария", "Сергей", "Ольга", "Дмитрий", "Анна"]
LAST_RU = ["Иванов", "Петров", "Сидоров", "Кузнецова", "Смирнов"]
RU_SUFFIX = ["ович", "овна"]
DOMAINS = ["mail.ru", "gmail.com", "inbox.kz", "bk.ru"]

PHONE_CODES = ["700", "701", "702", "705", "707", "747", "771", "775", "776", "777", "778"]


def gen_name(rng) -> str:
    if rng.random() < 0.5:  # казахское имя с отчеством-суффиксом
        return f"{rng.choice(LAST_KZ)} {rng.choice(FIRST_KZ)} {rng.choice(KZ_SUFFIX)}"
    return f"{rng.choice(LAST_RU)} {rng.choice(FIRST_RU)} {rng.choice(RU_SUFFIX)}"


def gen_phone(rng) -> str:
    code = rng.choice(PHONE_CODES)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(7))
    tmpl = rng.choice([
        f"+7 {code} {digits[:3]} {digits[3:5]} {digits[5:]}",
        f"8{code}{digits[:3]}{digits[3:5]}{digits[5:]}",
        f"8 ({code}) {digits[:3]}-{digits[3:5]}-{digits[5:]}",
        f"+7{code}{digits}",
    ])
    return tmpl


def gen_email(rng) -> str:
    name = rng.choice(FIRST_RU + FIRST_KZ).lower().replace("і", "i").replace("ә", "a")
    return f"{name}.{rng.randint(10, 99)}@{rng.choice(DOMAINS)}"


def gen_passport_kz(rng) -> str:
    return rng.choice([
        "".join(str(rng.randint(0, 9)) for _ in range(9)),                      # внутренний
        rng.choice("ABCDEMHNPR") + "".join(str(rng.randint(0, 9)) for _ in range(7)),  # загран
    ])


# «похожие, но не ПДн» — для негативных примеров
def gen_loan_number(rng) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(10))


def gen_iin_bad_checksum(rng) -> str:
    for _ in range(50):
        iin = "".join(str(rng.randint(0, 9)) for _ in range(12))
        if not validate_iin(iin):
            return iin
    return "123456789012"


def gen_iban_bad_checksum(rng) -> str:
    iban = random_iban(rng=rng)
    return iban[:-1] + str((int(iban[-1]) + 1) % 10)


def gen_date_like(rng) -> str:
    return f"{rng.randint(1, 28):02d}{rng.randint(1, 12):02d}{rng.randint(10, 26):02d}"


PLACEHOLDERS = {
    "iin": lambda rng: generate_iin(rng=rng),
    "bin": lambda rng: generate_bin(rng=rng),
    "iban": lambda rng: random_iban(rng=rng),
    "card": lambda rng: generate_card(rng=rng),
    "phone": gen_phone,
    "email": gen_email,
    "passport": gen_passport_kz,
    "name": gen_name,
}

# негативные тела: только «двойники» ПДн, ни одного span
NEGATIVE_BODIES = [
    "договор займа № {loan} от {date}",
    "счёт на оплату № {loan}, сумма по договору {loan}",
    "заявка {loan} обработана, ждём верификацию",
    "л/с {loan}, показания за {date}",
]


def build_body(body_tmpl: str, rng) -> tuple[str, list[Span], list[str]]:
    """Возвращает (текст тела, спаны, применённый нойз)."""
    buf = ""
    spans: list[Span] = []
    rest = body_tmpl
    while rest:
        head, sep, tail = rest.partition("{")
        buf += head
        if not sep:
            break
        key, sep, rest = tail.partition("}")
        assert sep, f"unbalanced braces in {body_tmpl!r}"
        value = PLACEHOLDERS[key](rng)
        start = len(buf)
        buf += value
        spans.append(Span(start=start, end=start + len(value), type=f"PLACEHOLDER:{key}", value=value))
    return buf, spans


def finalize_spans(spans: list[Span]) -> list[Span]:
    """Отображаем placeholder-типы в типы детекторов."""
    mapping = {
        "iin": "KZ_IIN", "bin": "KZ_BIN", "iban": "KZ_IBAN", "card": "CARD",
        "phone": "PHONE", "email": "EMAIL", "passport": "KZ_PASSPORT", "name": "PERSON",
    }
    for s in spans:
        s.type = mapping[s.type.split(":")[-1]]
    return spans


def build_example(rng, idx, *, split: str, noise_kinds=("ocr",), id_prefix="kzpb-A") -> Example:
    ctx = rng.choice(list(CONTEXTS))
    tmpl = rng.choice(CONTEXTS[ctx])
    is_negative = rng.random() < 0.25

    sep = rng.choice([". ", " — ", ": ", "\n"])
    prefix = tmpl + sep
    if is_negative:
        # тело без ПДн: нойз применяем к «двойникам», спанов нет
        body = rng.choice(NEGATIVE_BODIES).format(
            loan=gen_loan_number(rng),
            date=gen_date_like(rng),
        )
        spans: list[Span] = []
        noise_applied = []
        if rng.random() < 0.3:
            kind = rng.choice(noise_kinds)
            body = "".join(apply_noise_to_span(body, i, i + 1, kind, rng) for i in range(len(body)))
            noise_applied.append(kind)
    else:
        body, spans = build_body(rng.choice(BODIES), rng)
        noise_applied = []
        if rng.random() < 0.5 and spans:
            kind = rng.choice(noise_kinds)
            for s in spans:
                body = apply_noise_to_span(body, s.start, s.end, kind, rng)
            noise_applied.append(kind)
        spans = finalize_spans(spans)
        # после нойза значение в тексте изменилось — перечитываем ground truth из текста
        for s in spans:
            s.value = body[s.start:s.end]

    # офсеты тела → офсеты полного текста
    for s in spans:
        s.start += len(prefix)
        s.end += len(prefix)
    text = prefix + body

    ex = Example(
        id=f"{id_prefix}{idx:06d}",
        text=text,
        spans=spans,
        layer="A",
        context=ctx,
        noise=noise_applied,
        lang=LANG_BY_CONTEXT.get(ctx, "ru"),
        split=split,
        generator="kz-identifiers/0.1.0",
    )
    return ex


def split_for(seed: int, idx: int, test_ratio: float = 0.2) -> str:
    h = hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest()
    return "test" if int(h[:8], 16) / 0xFFFFFFFF < test_ratio else "dev"


def generate(n: int, seed: int, test_ratio: float = 0.2, noise_kinds=("ocr",)):
    dev: list[Example] = []
    test: list[Example] = []
    rng = random.Random(seed)
    for idx in range(n):
        split = split_for(seed, idx, test_ratio)
        ex = build_example(rng, idx, split=split, noise_kinds=noise_kinds)
        (test if split == "test" else dev).append(ex)
    return dev, test
