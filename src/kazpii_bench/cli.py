"""CLI:python -m kazpii_bench build --layer A --n 5000 --seed 42 --dev-out ... --test-out ..."""
from __future__ import annotations

import argparse
import json
import sys

from .layer_a import generate
from .schema import file_sha256, write_jsonl


def build_cmd(args) -> int:
    noise_kinds = args.noise.split(",") if args.noise else ("ocr",)
    dev, test = generate(args.n, seed=args.seed, test_ratio=args.test_ratio, noise_kinds=noise_kinds)
    n_dev = write_jsonl((e for e in dev), args.dev_out)
    n_test = write_jsonl((e for e in test), args.test_out)
    manifest = {
        "schema_version": "0.1.0",
        "layer": "A",
        "seed": args.seed,
        "n_requested": args.n,
        "n_dev": n_dev,
        "n_test": n_test,
        "test_ratio": args.test_ratio,
        "noise": list(noise_kinds),
        "generator": "kz-identifiers/0.1.0",
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kazpii-bench", description="KazPII-Bench dataset builder")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="сгенерировать слой датасета")
    b.add_argument("--layer", default="A", choices=["A"], help="пока только слой A")
    b.add_argument("--n", type=int, default=5000)
    b.add_argument("--seed", type=int, default=20261011)
    b.add_argument("--test-ratio", type=float, default=0.2)
    b.add_argument("--noise", default="ocr", help="список нойзов через запятую: ocr,kaz_loose,translit")
    b.add_argument("--dev-out", required=True)
    b.add_argument("--test-out", required=True)
    b.add_argument("--manifest", default="")
    b.set_defaults(func=build_cmd)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
