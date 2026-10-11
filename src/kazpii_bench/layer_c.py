"""Слой C: перефразирование семействами LLM через OpenRouter.

Принцип «генератор ≠ система ≠ судья»: каждая модель-генератор переписывает
текст, сохраняя идентификаторы дословно внутри маркеров. Верификация
детерминированная (без LLM-судьи): маркеры на месте, значения байт-в-байт те же,
офсеты пересчитаны из позиций маркеров. Нарушение → пример отклоняется и идёт
в статистику отказов генератора.

Стоимость (проверено на 3000 примерах): ~$1 за модель, всего ~$3 на три.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass

from .schema import Example, Span

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Зафиксированные дешёвые модели из РАЗНЫХ семейств (проверено 2026-10-11).
# Цены за 1M токенов: luna $0.1/0.5, haiku $0.1/0.5, deepseek $0.3/1.2.
MODELS = {
    "openai/gpt-6-luna": "openai",
    "anthropic/claude-haiku-5.5": "anthropic",
    "deepseek/deepseek-v4.1-flash": "deepseek",
}

TYPES = ["KZ_IIN", "KZ_BIN", "KZ_IBAN", "KZ_PASSPORT", "CARD", "PHONE", "EMAIL", "PERSON"]

STYLES = ["whatsapp", "memo", "contract", "kk_chat"]

STYLE_PROMPT = {
    "whatsapp": "неформальное сообщение в мессенджере, короткие фразы",
    "memo": "служебная записка, сухой деловой стиль",
    "contract": "фрагмент договора, канцелярные обороты",
    "kk_chat": "сообщение в мессенджере с казахскими словами и именами",
}

MARKER_RE = re.compile(r"⟨(KZ_IIN|KZ_BIN|KZ_IBAN|KZ_PASSPORT|CARD|PHONE|EMAIL|PERSON)⟩(.*?)⟨/\1⟩", re.S)


@dataclass
class ParaphraseResult:
    text: str
    spans: list[Span]
    accepted: bool
    reason: str = ""


def _post_chat(model: str, prompt: str, api_key: str, timeout: float = 60.0) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1200,
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/k1ddy/kazpii-bench",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"].strip()


def build_prompt(text: str, style: str) -> str:
    return (
        f"Перепиши следующий текст в стиле «{STYLE_PROMPT[style]}». "
        "Сохрани смысл, порядок мыслей и все маркеры ⟨...⟩ с их содержимым "
        "ДОСЛОВНО, включая все цифры и буквы внутри. Не меняй ничего внутри "
        "маркеров, не добавляй и не удаляй маркеры. Верни только текст.\n\n"
        f"{text}"
    )


def mark_text(text: str, spans: list[Span]) -> str:
    """Оборачиваем значения в маркеры — по ним модель не имеет права менять."""
    out, last = [], 0
    for s in sorted(spans, key=lambda x: x.start):
        out.append(text[last : s.start])
        out.append(f"⟨{s.type}⟩{text[s.start : s.end]}⟨/{s.type}⟩")
        last = s.end
    out.append(text[last:])
    return "".join(out)


def verify_paraphrase(original: str, spans: list[Span], rewritten: str) -> ParaphraseResult:
    """Детерминированная проверка: маркеры целы, значения те же, новыe офсеты."""
    found = MARKER_RE.findall(rewritten)
    if not found:
        return ParaphraseResult(rewritten, [], False, "no_markers")

    expect = {(s.type, s.value) for s in spans}
    got = {(t, v) for t, v in found}
    if expect != got:
        return ParaphraseResult(rewritten, [], False, "values_changed")

    plain = MARKER_RE.sub(lambda m: m.group(2), rewritten)
    new_spans, pos = [], 0
    for m in MARKER_RE.finditer(rewritten):
        t, v = m.group(1), m.group(2)
        idx = plain.find(v, pos)
        if idx < 0:
            return ParaphraseResult(rewritten, [], False, "offset_rebuild_failed")
        new_spans.append(Span(start=idx, end=idx + len(v), type=t, value=v))
        pos = idx + len(v)
    for s in new_spans:
        assert plain[s.start : s.end] == s.value
    return ParaphraseResult(plain, new_spans, True)


def paraphrase_example(
    rec: dict, model: str, api_key: str, style: str | None = None
) -> tuple[Example | None, str]:
    """Возвращает (Example|None, reason). reason != '' если отклонён."""
    rng = random.Random(hash((rec["id"], model)) & 0xFFFF)
    style = style or rng.choice(STYLES)
    spans = [
        Span(start=s["start"], end=s["end"], type=s["type"], value=s["value"]) for s in rec.get("spans", [])
    ]
    if not spans:
        return None, "no_spans"
    marked = mark_text(rec["text"], spans)
    try:
        out = _post_chat(model, build_prompt(marked, style), api_key)
    except Exception as e:
        return None, f"api_error:{type(e).__name__}"
    res = verify_paraphrase(rec["text"], spans, out)
    if not res.accepted:
        return None, res.reason
    ex = Example(
        id=rec["id"].replace("kzpb-", "kzpbC-") + f"-{model.split('/')[-1][:12]}",
        text=res.text,
        spans=res.spans,
        layer="C",
        generator=model,
        context=style,
        lang=rec.get("lang", "ru"),
        split=rec.get("split", "dev"),
    )
    return ex, ""


def total_cost_estimate(n: int, avg_in: int = 320, avg_out: int = 420) -> dict:
    """Оценка стоимости прогона: цены за 1M токенов из MODELS_PRICING."""
    pricing = {
        "openai/gpt-6-luna": (0.10, 0.50),
        "anthropic/claude-haiku-5.5": (0.100, 0.500),
        "deepseek/deepseek-v4.1-flash": (0.300, 1.200),
    }
    return {
        m: round((n * avg_in * pin / 1e6) + (n * avg_out * pout / 1e6), 3)
        for m, (pin, pout) in pricing.items()
    }


def build_layer_c(seed_path, models, n, seed=20261011, test_ratio=0.2):
    """Прогон слоя C: читает seed-набор, перефразирует, верифицирует кодом."""
    import json as _json

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY required (см. .env phase2)")
    recs = []
    with open(seed_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(_json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(recs)
    recs = recs[: n * len(models)]

    dev, test, stats = [], [], {"accepted": 0, "rejected": {}}
    for i, rec in enumerate(recs):
        model = models[i % len(models)]
        ex, reason = paraphrase_example(rec, model, api_key)
        if ex is None:
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
            continue
        stats["accepted"] += 1
        (test if ex.split == "test" else dev).append(ex)
    stats["total"] = stats["accepted"] + sum(stats["rejected"].values())
    stats["accept_rate"] = round(stats["accepted"] / stats["total"], 4) if stats["total"] else None
    stats["cost_estimate_usd"] = total_cost_estimate(stats["accepted"])
    return dev, test, stats
