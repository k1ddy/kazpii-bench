"""Нойзы: подстановки строго 1:1 (один символ → один символ).

Правило инварианта: любая замена сохраняет длину строки, поэтому офсеты
спанов не сдвигаются. Тест это проверяет для всех карт замен.
"""

from __future__ import annotations

#: OCR-шум: кириллица/латиница/цифры, которые путает OCR
OCR_MAP = {
    "0": "О",
    "1": "l",
    "3": "З",
    "5": "Б",
    "6": "б",
    "о": "0",
    "з": "3",
    "б": "6",
    "л": "l",
}

#: «бытовая» замена казахских букв в неформальной переписке
KAZ_LOOSE_MAP = {"қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "h", "і": "i", "ә": "а"}

#: транслитерация для смешения ru/kk
TRANSLIT = {"И": "I", "і": "i", "һ": "h", "қ": "k", "ң": "n", "ө": "o", "ұ": "u", "ү": "y", "ә": "a"}

NOISE_MAPS = {"ocr": OCR_MAP, "kaz_loose": KAZ_LOOSE_MAP, "translit": TRANSLIT}


def apply_noise(text: str, chars: str, noise: str, rng) -> str:
    """Применить нойз к конкретным символам строки (1:1 подстановки)."""
    table = NOISE_MAPS[noise]
    out = list(text)
    for i in chars:
        c = text[i]
        if c in table and rng.random() < 0.6:  # не каждый экземпляр — как в жизни
            out[i] = table[c]
    return "".join(out)


def apply_noise_to_span(text: str, start: int, end: int, noise: str, rng) -> str:
    return apply_noise(text, range(start, end), noise, rng)
