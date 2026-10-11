"""Тесты слоя C без сети: мок OpenRouter + проверка инвариант."""

from __future__ import annotations

from kazpii_bench.layer_c import (
    MODELS,
    build_prompt,
    mark_text,
    total_cost_estimate,
    verify_paraphrase,
)
from kazpii_bench.schema import Span

TEXT = "оплата на KZ86125KZT5004100100, ИИН 061211600012"
SPANS = [
    Span(start=10, end=30, type="KZ_IBAN", value="KZ86125KZT5004100100"),
    Span(start=36, end=48, type="KZ_IIN", value="061211600012"),
]


def test_mark_text_preserves_values_and_offsets():
    marked = mark_text(TEXT, SPANS)
    for s in SPANS:
        assert f"⟨{s.type}⟩{s.value}⟨/{s.type}⟩" in marked


def test_accept_good_paraphrase():
    rewritten = (
        "переведи, пожалуйста, оплату на ⟨KZ_IBAN⟩KZ86125KZT5004100100⟨/KZ_IBAN⟩. "
        "Мой ИИН: ⟨KZ_IIN⟩061211600012⟨/KZ_IIN⟩, спасибо."
    )
    res = verify_paraphrase(TEXT, SPANS, rewritten)
    assert res.accepted, res.reason
    assert len(res.spans) == 2
    for s in res.spans:
        assert res.text[s.start : s.end] == s.value


def test_reject_changed_value():
    rewritten = "переведи на ⟨KZ_IBAN⟩KZ86125KZT5004100199⟨/KZ_IBAN⟩, ИИН ⟨KZ_IIN⟩061211600012⟨/KZ_IIN⟩"
    assert verify_paraphrase(TEXT, SPANS, rewritten).reason == "values_changed"


def test_reject_missing_marker():
    assert (
        verify_paraphrase(TEXT, SPANS, "переведи на KZ86125KZT5004100100, ИИН 061211600012").reason
        == "no_markers"
    )


def test_reject_dropped_marker():
    rewritten = "переведи на ⟨KZ_IBAN⟩KZ86125KZT5004100100⟨/KZ_IBAN⟩, ИИН 061211600012"
    assert verify_paraphrase(TEXT, SPANS, rewritten).reason == "values_changed"


def test_pinned_models_are_three_families():
    families = set(MODELS.values())
    assert len(MODELS) == 3 and len(families) == 3, "генераторы должны быть из разных семейств"


def test_cost_estimate_is_cents_not_dollars_per_example():
    est = total_cost_estimate(3000)
    assert set(est) == set(MODELS)
    assert max(est.values()) < 5.0, "прогон 3000×3 должен стоить единицы долларов"


def test_prompt_forbids_touching_markers():
    p = build_prompt(mark_text(TEXT, SPANS), "whatsapp")
    assert "ДОСЛОВНО" in p and "маркеров" in p
