# KazPII-Bench

Открытый бенчмарк для оценки детекторов персональных данных в казахстанских и
русских текстах. Цель — дать публичную воспроизводимую меру качества, которой
сейчас нет на рынке: коммерчески чистого казахского PII-детектора с проверенными
метриками не существует.

Open benchmark for PII detection in Kazakh and Russian texts. The gap it fills:
there is no public, reproducible way to measure how systems find Kazakhstani PII.

## Слои датасета

| Слой | Источник | Статус |
|---|---|---|
| A | Программная генерация: kz-identifiers × контексты × OCR-шумы | **готов, 5000 примеров** |
| B | KazNERD (112 702 предложения, CC BY 4.0) с доразметкой | **готов, 2390 примеров** |
| C | Перефразирование 3 семействами LLM (gpt-6-luna / haiku-5.5 / deepseek-v4.1-flash) | **готов: 4913 примеров** |
| D | Золотое подмножество, двойная ручная разметка | адъюдикация границ, не начат |

Тестовая часть (20 %) закрыта: хранится вне репозитория, публикуются только
метрики и лидерборд. Dev-часть открыта.

## Схема (JSONL)

```json
{
  "id": "kzpb-A000001",
  "text": "привет, скинь реквизиты — ФИО: Иванов Иван ович, ИИН 900517301146",
  "spans": [{"start": 41, "end": 57, "type": "KZ_IIN", "value": "900517301146"}],
  "layer": "A",
  "synthetic": true,
  "generator": "kz-identifiers/0.1.0",
  "context": "whatsapp",
  "noise": ["ocr"],
  "lang": "ru",
  "split": "dev",
  "canary": "kzpii-bench-canary-2026",
  "schema_version": "0.1.0"
}
```

Типы спанов соответствуют продакшен-детектору шлюза: `KZ_IIN`, `KZ_BIN`,
`KZ_IBAN`, `KZ_PASSPORT`, `CARD`, `PHONE`, `EMAIL`, `PERSON`, `ORG`, `LOC`.

Инварианты (покрыты тестами):

- `text[start:end] == value` — офсеты всегда точные;
- нойз только 1:1 подстановками — офсеты не сдвигаются;
- генерация детерминирована по seed;
- каждые запись несёт canary-строку для детекции утечки в обучение;
- `split` по SHA-256 — dev/test не пересекаются.

## Воспроизведение замеров

Лёгкий набор (CI): `pip install -e ".[dev]" && pytest -m "not crosscheck"`.

Тяжёлые детекторы ставятся отдельно, вне CI:

```bash
pip install presidio-analyzer natasha gliner torch
# presidio дополнительно тянет spaCy-модель en_core_web_lg при первом запуске
```

Замеры из `results/`:

```bash
python -m kazpii_bench eval --detector baseline --dataset data/samples/dev_sample_5000.jsonl --mode span
python -m kazpii_bench eval --detector gateway   --dataset data/samples/dev_sample_5000.jsonl --mode value --url http://127.0.0.1:18451
python -m kazpii_bench eval --detector presidio  --dataset data/samples/dev_sample_5000.jsonl --mode span
python -m kazpii_bench eval --detector natasha   --dataset data/samples/dev_sample_5000.jsonl --mode span
python -m kazpii_bench eval --detector gliner    --dataset data/samples/dev_sample_5000.jsonl --mode span
```

Замечания по воспроизводимости: GLiNER требует `transformers==4.53.x`
(на 5.x ломается токенайзер) и `protobuf`, `tiktoken`, `sentencepiece`.

### Что показал слой C (см. `results/`)

- **Структурные идентификаторы не обобщаются плохо — они не обобщаются вообще:**
  парное сравнение «оригинал ↔ перефраз» (n=2347) даёт падение recall
  baseline **0.000** — regex+checksum находят значение в любом обрамлении.
- **NER/контекстная детекция — другое дело:** на перефразах шлюз даёт
  PERSON recall 0.10 при **1363 ложных срабатываниях** (на шаблонах было 89),
  overall F1 падает 0.78 → 0.50. Смысл слоя C — ровно здесь: без него бенчмарк
  мерил бы только 3 типа из 9 и не видел бы главной слабости продукта.
- **Отказы генераторов:** настоящий провал модели — 2 из 2969 у deepseek
  (после отключения reasoning); остальные «отказы» — seed-записи без ПДн.
- **Аудит-след:** каждый вызов пишется с промтом, ответом и finish_reason
  (`--audit-out`), чтобы сверки шли оффлайн. Для deepseek-прогона аудит
  сохранён; для первого прогона luna/haiku — нет (баг проброса аргумента,
  исправлен, тест на аудит добавлен).

## Генерация

```bash
pip install -e ".[dev]"
python -m kazpii_bench build --n 5000 --seed 20261011 \
    --dev-out data/dev.jsonl --test-out data/private/test.jsonl \
    --manifest data/manifest.json
```

Шум (`--noise`): `ocr` — путаница OCR (0↔О, 1↔l, 3↔З), `kaz_loose` — бытовая
замена казахских букв (қ→к), `translit` — смешение латиницы/кириллицы.
Комбинируется через запятую.

### Слой C (перефраз)

```bash
export OPENROUTER_API_KEY=sk-or-...
python -m kazpii_bench build --layer C --models openai/gpt-6-luna,anthropic/claude-haiku-5.5,deepseek/deepseek-v4.1-flash \
    --n 3000 --dev-in data/samples/dev_sample_5000.jsonl \
    --dev-out data/dev_C.jsonl --test-out data/private/test_C.jsonl
```

Модели закреплены в `layer_c.py:MODELS` (проверено 2026-10-11: три семейства,
~$3 за прогон 3000×3). Верификация детерминированная: значения внутри маркеров
`⟨TYPE⟩…⟨/TYPE⟩` обязаны совпасть байт-в-байт, иначе пример отклоняется и
попадает в статистику отказов генератора.

Негативные примеры (25 %): номера договоров и займов из 10 цифр, ИИН с
невалидной checksum, даты в виде чисел, IBAN с битым MOD-97 — ни одного спана.
Они измеряют ложные срабатывания.

## Почему генератор ≠ система ≠ судья

LLM-оценщик «scores its own outputs higher than others'» (Panickssery, Bowman,
Feng, NeurIPS 2024). Поэтому генератор слоя C, проверяемая система и судья
берутся из разных семейств моделей, а метрики публикуются в разбивке по
генератору. Синтетика подтверждает механику, но не смысл: проверять реализацию
надо против спецификаций, независимых реализаций и реальных номеров.

## Первый замер (см. `results/leaderboard.md`)

dev-выборка 400, режимы: baseline/presidio/natasha/gliner — span, gateway — value:

| детектор | P clean | R clean | F1 clean | F1 | ocr | FP на негативах |
|---|---|---|---|---|---|
| baseline (regex+checksum) | 1.000 | 0.828 | 0.906 | 0.181 | 0/104 |
| gateway (наш шлюз) | 0.792 | 0.768 | 0.779 | 0.151 | 20/104 |
| presidio | 0.133 | 0.337 | 0.191 | 0.113 | 4287/104 |
| gliner (zero-shot) | 0.092 | 0.051 | 0.065 | 0.041 | 33/104 |
| natasha | 0.067 | 0.007 | 0.012 | 0.000 | 11/104 |

Вывод: ни один opensource-детектор из коробки не находит ИИН/БИН/паспорт РК;
OCR-шум валит всех. Это и есть публичная «измеримость», которой не было.

## Статус верификации

- [x] ИИН/БИН/IBAN/карты — алгоритмы сверены с adilet V2300032918, SWIFT
      registry и двумя независимыми реализациями (kz-identifiers 0.1.0).
- [x] Слой B — KazNERD, CC BY 4.0, атрибуция ISSAI в схеме записи.
- [ ] Слои C/D — не начаты.

## Лицензия

Код — Apache-2.0. Данные слоёв B–D — CC BY 4.0 с атрибуцией источников
(см. `datasheet.md`).

## Методология и честные границы

Подробности — `results/leaderboard.md`, раздел «Оговорки». Коротко:

- baseline использует те же алгоритмы, что и таксономия бенча, — это планка,
  не конкурент;
- Presidio даёт thousands детекций вне таксономии (US_*, URL, NRP) — its
  precision надо цитировать с оговоркой;
- GLiнер zero-shot на английских метках путает классы (в пробе ИИН → email);
- gateway сравнивался в value-режиме, остальные — в span;
- n=400, тестовая часть закрыта, все прогоны воспроизводимы командами.
