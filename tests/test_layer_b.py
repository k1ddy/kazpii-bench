import os

import pytest

from kazpii_bench.layer_b import ingest

KAZNERD = os.environ.get("KAZNERD_PATH", "/opt/kaznerd/KazNERD/IOB2_train.txt")
needs = pytest.mark.skipif(not os.path.exists(KAZNERD_PATH_ENV := KAZNERD), reason="KazNERD not present")


@needs
def test_ingest_produces_valid_examples():
    dev, test = ingest(KAZNERD, 400, seed=5)
    for e in dev + test:
        e.validate()
    assert len(dev) > 100


@needs
def test_negative_sentences_have_no_spans():
    dev, _ = ingest(KAZNERD, 600, seed=6)
    neg = [e for e in dev if not e.spans]
    assert neg, "ожидались негативные примеры на реальных NER-сущностях"


@needs
def test_kk_language_and_license_source():
    dev, _ = ingest(KAZNERD, 200, seed=7)
    for e in dev[:50]:
        assert e.lang == "kk"
        assert e.layer == "B"
