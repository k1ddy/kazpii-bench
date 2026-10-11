"""CLI:python -m kazpii_bench build --layer A --n 5000 --seed 42 --dev-out ... --test-out ..."""

from __future__ import annotations

import argparse
import json
import sys

from .harness import evaluate as run_evaluate
from .layer_a import generate
from .layer_b import ingest
from .schema import file_sha256, write_jsonl


def build_cmd(args) -> int:
    if args.layer == "A":
        noise_kinds = tuple(args.noise.split(",")) if args.noise else ("ocr",)
        dev, test = generate(args.n, seed=args.seed, test_ratio=args.test_ratio, noise_kinds=noise_kinds)
        generator = "kz-identifiers/0.1.0"
    else:
        if not args.kaznerd:
            print("--kaznerd required for layer B", file=sys.stderr)
            return 2
        dev, test = ingest(args.kaznerd, args.n, seed=args.seed, test_ratio=args.test_ratio)
        generator = "kz-identifiers/0.1.0 + KazNERD CC BY 4.0"
    n_dev = write_jsonl((e for e in dev), args.dev_out)
    n_test = write_jsonl((e for e in test), args.test_out)
    manifest = {
        "schema_version": "0.1.0",
        "layer": args.layer,
        "seed": args.seed,
        "n_requested": args.n,
        "n_dev": n_dev,
        "n_test": n_test,
        "test_ratio": args.test_ratio,
        "noise": list(locals().get("noise_kinds", [])),
        "generator": generator,
        "files": {
            "dev": {"path": args.dev_out, "sha256": file_sha256(args.dev_out)},
            "test": {"path": args.test_out, "sha256": file_sha256(args.test_out)},
        },
        "note": "test-часть закрыта: хранится вне публичного репозитория",
    }
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, ensure_ascii=False))
    print(f"dev -> {args.dev_out} ({n_dev}), test -> {args.test_out} ({n_test})")
    return 0


def eval_cmd(args) -> int:
    if args.detector == "baseline":
        from .detectors import BaselineDetector

        det = BaselineDetector()
    elif args.detector == "gateway":
        from .detectors import GatewayScanDetector

        det = GatewayScanDetector(args.url)
    elif args.detector == "presidio":
        from .detectors_ext import PresidioDetector

        det = PresidioDetector()
    elif args.detector == "natasha":
        from .detectors_ext import NatashaDetector

        det = NatashaDetector()
    else:
        from .detectors_ext import GlinerDetector

        det = GlinerDetector()
    mode = "value" if args.detector == "gateway" else args.mode
    result = run_evaluate(args.dataset, det.detect, mode=mode)
    from .harness import report_markdown

    print(report_markdown(result))
    if args.out:
        import json as _json

        with open(args.out, "w", encoding="utf-8") as f:
            _json.dump(result, f, ensure_ascii=False, indent=2)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kazpii-bench", description="KazPII-Bench dataset builder")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="сгенерировать слой датасета")
    b.add_argument("--layer", default="A", choices=["A", "B"], help="A — синтетика, B — KazNERD+вставка")
    b.add_argument("--kaznerd", default="", help="путь к IOB2_train.txt KazNERD (для слоя B)")
    b.add_argument("--n", type=int, default=5000)
    b.add_argument("--seed", type=int, default=20261011)
    b.add_argument("--test-ratio", type=float, default=0.2)
    b.add_argument("--noise", default="ocr", help="список нойзов через запятую: ocr,kaz_loose,translit")
    b.add_argument("--dev-out", required=True)
    b.add_argument("--test-out", required=True)
    b.add_argument("--manifest", default="")
    b.set_defaults(func=build_cmd)

    e = sub.add_parser("eval", help="прогнать детектор по датасету")
    e.add_argument(
        "--detector", required=True, choices=["baseline", "gateway", "presidio", "natasha", "gliner"]
    )
    e.add_argument("--dataset", required=True)
    e.add_argument("--url", default="http://127.0.0.1:18451", help="база шлюза для detector=gateway")
    e.add_argument("--mode", default="value", choices=["span", "value"])
    e.add_argument("--out", default="")
    e.set_defaults(func=eval_cmd)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
