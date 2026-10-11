from kazpii_bench.detectors import BaselineDetector
from kazpii_bench.harness import (
    canon_type,
    match_spans,
    match_values,
    sig,
)


def test_sig_ignores_case_and_separators():
    assert sig("+7 701 555-12-34") == sig("+77015551234")
    assert sig("alice.74@inbox.kz") == sig("ALICE.74@INBOX.KZ")


def test_canon_type_maps_ner_aliases():
    assert canon_type("RU_PERSON") == "PERSON"
    assert canon_type("KZ_IIN") == "KZ_IIN"


def test_match_values_firstlast():
    gold = [{"type": "KZ_IIN", "start": 0, "end": 12, "value": "900517301146"}]
    det = [type("D", (), {"type": "KZ_IIN", "value": "90|46", "start": None, "end": None})()]
    tp, fp, fn = match_values(gold, det)
    assert (tp, fp, fn) == (1, 0, 0)


def test_match_spans_exact():
    gold = [{"type": "CARD", "start": 5, "end": 21, "value": "x" * 16}]
    det = [type("D", (), {"type": "CARD", "value": "", "start": 5, "end": 21})()]
    tp, fp, fn = match_spans(gold, det)
    assert (tp, fp, fn) == (1, 0, 0)
    det2 = [type("D", (), {"type": "CARD", "value": "", "start": 6, "end": 21})()]
    assert match_spans(gold, det2) == (0, 1, 1)


def test_baseline_detects_clean_iin():
    d = BaselineDetector()
    got = d.detect("ИИН 900517301146 и тел +7 701 555 12 34")
    types = {x.type for x in got}
    assert "KZ_IIN" in types and "PHONE" in types
    for x in got:
        assert "ИИН 900517301146 и тел +7 701 555 12 34"[x.start:x.end]


def test_negatives_produce_zero_findings():
    d = BaselineDetector()
    assert d.detect("договор займа № 100238471582 от 12.03") == []
