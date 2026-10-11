"""Харнес KazPII-Bench: прогон детекторов и метрики.

Два режима сопоставления:
  - span  — точный матч (start, end, type) для детекторов с офсетами;
  - value — матч (type, нормализованное значение) для HTTP-детекторов,
           которые возвращают плейсхолдеры вместо офсетов (шлюз /redact).

Метрики: микро и макро precision/recall/F1, в том числе по типам;
отдельный отчёт по негативным примерам ложные срабатывания.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Detected:
    type: str
    value: str
    start: int | None = None
    end: int | None = None


TYPE_ALIASES = {"RU_PERSON": "PERSON", "RU_ORG": "ORG", "RU_LOCATION": "LOC"}


def canon_type(t: str) -> str:
    return TYPE_ALIASES.get(t, t)


def norm_value(v: str) -> str:
    return re.sub(r"[\s\-()]", "", v).upper()


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_type: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0]))  # tp,fp,fn
    fp_on_negative: int = 0
    negatives: int = 0

    def prf(self):
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f


def match_spans(gold: list[dict], det: list[Detected]) -> tuple[int, int, int]:
    """Точный матч: (type, start, end). Возвращает (tp, fp, fn)."""
    gold_set = {(g["type"], g["start"], g["end"]) for g in gold}
    det_set = {(d.type, d.start, d.end) for d in det if d.start is not None}
    tp = len(gold_set & det_set)
    fp = len(det_set - gold_set)
    fn = len(gold_set - det_set)
    return tp, fp, fn


def sig(v: str) -> str:
    """Сигнатура first2+last2 — устойчива к FPE-токенам и sample-маскам."""
    n = norm_value(v)
    return n[:2] + n[-2:]


def match_values(gold: list[dict], det: list[Detected]) -> tuple[int, int, int]:
    """Матч по (type, first2+last2) — устойчив к sample-маскам и FPE-токенам."""
    gold_set = {(canon_type(g["type"]), sig(g["value"])) for g in gold}
    det_set = {(canon_type(d.type), sig(d.value)) for d in det}
    tp = len(gold_set & det_set)
    fp = len(det_set - gold_set)
    fn = len(gold_set - det_set)
    return tp, fp, fn


def evaluate(dataset_path, detect_fn, mode: str = "span") -> dict:
    errors = []
    m = Metrics()
    by_noise: dict = defaultdict(Metrics)
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            try:
                det = detect_fn(rec["text"])
            except Exception as e:  # детектор упал — считаем все пропущенным
                det = []
                errors.append({"id": rec["id"], "error": str(e)[:200]})
            gold = rec["spans"]
            if not gold:
                m.negatives += 1
                m.fp_on_negative += len(det)
                continue
            if mode == "span":
                tp, fp, fn = match_spans(gold, det)
            else:
                tp, fp, fn = match_values(gold, det)
            m.tp += tp
            m.fp += fp
            m.fn += fn
            noise_key = ",".join(rec.get("noise") or []) or "clean"
            nb = by_noise[noise_key]
            nb.tp += tp
            nb.fp += fp
            nb.fn += fn
            # по типам — тем же матчингом, что и micro
            gold_by_type = defaultdict(set)
            det_by_type = defaultdict(set)
            if mode == "span":
                for g in gold:
                    gold_by_type[canon_type(g["type"])].add((canon_type(g["type"]), g["start"], g["end"]))
                for d in det:
                    if d.start is not None:
                        det_by_type[canon_type(d.type)].add((canon_type(d.type), d.start, d.end))
            else:
                for g in gold:
                    gold_by_type[canon_type(g["type"])].add((canon_type(g["type"]), sig(g["value"])))
                for d in det:
                    det_by_type[canon_type(d.type)].add((canon_type(d.type), sig(d.value)))
            for t, gset in gold_by_type.items():
                dset = det_by_type.get(t, set())
                m.per_type[canon_type(t)][0] += len(gset & dset)
                m.per_type[canon_type(t)][2] += len(gset - dset)
            for t, dset in det_by_type.items():
                gset = gold_by_type.get(t, set())
                m.per_type[canon_type(t)][1] += len(dset - gset)

    p, r, f = m.prf()
    return {
        "dataset": dataset_path,
        "mode": mode,
        "micro": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                  "tp": m.tp, "fp": m.fp, "fn": m.fn},
        "negatives": {"count": m.negatives, "false_positives": m.fp_on_negative,
                      "fp_rate": round(m.fp_on_negative / m.negatives, 4) if m.negatives else None},
        "per_type": {t: {"tp": v[0], "fp": v[1], "fn": v[2],
                         "recall": round(v[0] / (v[0] + v[2]), 4) if v[0] + v[2] else None}
                     for t, v in sorted(m.per_type.items())},
        "errors": errors[:20],
        "by_noise": {k: {"precision": round(v.prf()[0], 4), "recall": round(v.prf()[1], 4),
                         "f1": round(v.prf()[2], 4), "tp": v.tp, "fn": v.fn}
                     for k, v in sorted(by_noise.items())},
    }


def report_markdown(result: dict) -> str:
    lines = [f"# KazPII-Bench report — {result['dataset']}", ""]
    mi = result["micro"]
    lines += [f"- mode: `{result['mode']}`",
              f"- micro P/R/F1: **{mi['precision']:.3f} / {mi['recall']:.3f} / {mi['f1']:.3f}** "
              f"(tp={mi['tp']}, fp={mi['fp']}, fn={mi['fn']})"]
    neg = result["negatives"]
    lines += [f"- negatives: {neg['count']}, false positives: {neg['false_positives']} "
              f"(fp rate {neg['fp_rate']})", "",
              "| type | tp | fp | fn | recall |", "|---|---|---|---|---|"]
    for t, v in result["per_type"].items():
        lines.append(f"| {t} | {v['tp']} | {v['fp']} | {v['fn']} | {v['recall']} |")
    if result.get("by_noise"):
        lines += ["", "## По шуму", "", "| шум | P | R | F1 | tp | fn |", "|---|---|---|---|---|---|"]
        for k, v in result["by_noise"].items():
            lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} "
                         f"| {v['tp']} | {v['fn']} |")
    return "\n".join(lines)
