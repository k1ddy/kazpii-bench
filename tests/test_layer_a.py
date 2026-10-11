import json
import subprocess
import sys

from kazpii_bench import CANARY
from kazpii_bench.layer_a import generate
from kazpii_bench.noise import NOISE_MAPS, apply_noise


def test_offsets_and_values_exact():
    dev, test = generate(600, seed=42)
    for e in dev + test:
        e.validate()  # assert внутри


def test_determinism():
    a_dev, a_test = generate(120, seed=7)
    b_dev, b_test = generate(120, seed=7)
    assert [e.to_jsonl() for e in a_dev] == [e.to_jsonl() for e in b_dev]
    assert [e.to_jsonl() for e in a_test] == [e.to_jsonl() for e in b_test]


def test_split_disjoint_and_ratio():
    dev, test = generate(2000, seed=3, test_ratio=0.2)
    assert {e.id for e in dev} & {e.id for e in test} == set()
    assert abs(len(test) / (len(dev) + len(test)) - 0.2) < 0.03


def test_noise_is_length_preserving():
    import random

    rng = random.Random(0)
    text = "ИИН 900517301146, тел +7 701 555 12 34"
    for name in NOISE_MAPS:
        for _ in range(50):
            noised = apply_noise(text, range(len(text)), name, rng)
            assert len(noised) == len(text)


def test_negatives_have_no_spans():
    dev, _ = generate(1500, seed=11)
    neg = [e for e in dev if not e.spans]
    assert len(neg) > 50, "ожидали хотя бы 5% негативных примеров"


def test_canary_in_every_record():
    dev, _ = generate(200, seed=5)
    for e in dev:
        assert e.canary == CANARY


def test_type_coverage():
    dev, _ = generate(1200, seed=21)
    types = {s.type for e in dev for s in e.spans}
    assert {"KZ_IIN", "KZ_BIN", "KZ_IBAN", "CARD", "PHONE", "EMAIL", "KZ_PASSPORT", "PERSON"} <= types


def test_cli_writes_jsonl_and_manifest(tmp_path):
    dev_out = tmp_path / "dev.jsonl"
    test_out = tmp_path / "private" / "test.jsonl"
    man = tmp_path / "manifest.json"
    test_out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "kazpii_bench",
            "build",
            "--n",
            "300",
            "--seed",
            "9",
            "--dev-out",
            str(dev_out),
            "--test-out",
            str(test_out),
            "--manifest",
            str(man),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    lines = dev_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    rec = json.loads(lines[0])
    assert rec["layer"] == "A" and rec["canary"] == CANARY
    man_d = json.loads(man.read_text())
    assert man_d["n_dev"] + man_d["n_test"] == 300
