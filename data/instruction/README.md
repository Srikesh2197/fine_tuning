# Instruction data — provenance and design

## What this is

Synthetic instruction data in Alpaca format, consumed by notebook 02:

```json
{"instruction": "...", "input": "...", "output": "...", "kind": "qa", "topic": "model-risk"}
```

`input` is `""` for most records and carries a passage for excerpt and classification tasks.
`kind` and `topic` are metadata for analysis — **neither is shown to the model**.

Built by:

```bash
python scripts/generate_instructions.py
```

## In what sense is this "synthetic"?

Being precise about this, because "synthetic data" covers several quite different things:

- The ~120 QA **seeds are hand-authored** for this repository, grounded in the domain corpus.
- The **expansion is programmatic**: each seed is emitted twice, once bare and once through a
  request-form decorator ("Answer concisely: …", "For a new joiner on the platform team — …").
  This teaches the model that the same request arrives in different surface forms.
- The `subject` and `section` tasks are **derived mechanically** from `corpus.jsonl`, where
  document structure already supplies the correct answer.

This is the seed-and-expand pattern that self-instruct popularised, with the LLM expansion step
replaced by deterministic templating. The tradeoff: reproducible offline, zero cost, no API key —
but the paraphrases are mechanical rather than natural. `docs/concepts.md` covers what an
LLM-driven expansion would add and where it needs care (licence terms on training against another
model's outputs, and dedup against seed content).

## Composition

| kind | count | what it teaches |
|---|---|---|
| `qa` | 272 | Answer a domain question from parametric knowledge |
| `subject` | 18 | Classify which internal system a passage describes (short output) |
| `section` | 12 | Recover the section heading a passage sits under |
| `excerpt` | 12 | Reason over a supplied passage in the `input` field |
| `abstain` | 15 | **Decline** when the documentation doesn't cover the question |
| **total** | **329** | 295 train / 34 eval |

### Why abstention examples are in here

A model instruction-tuned on data where *every question has an answer* learns that every question
has an answer. It then answers confidently when it should decline — the single most damaging
failure mode for anything retrieval-adjacent. Abstention is ~4.4% of the training set. The corpus
itself argues for roughly one in ten; this set sits below that, which is a deliberate compromise
given the small total and is worth raising if you extend the data.

## The split is group-aware

Each QA seed produces two records with **identical outputs**. A naive random split would put the
bare form in train and the decorated form in eval, so eval loss would be measuring memorization.

The split therefore keys on an internal `_group` field (stripped before writing), so both
variants always land on the same side. It is also stratified by `kind` so every task type appears
in eval. Two guards run at generation time and again in `validate_data.py`:

1. No eval **prompt** appears in train.
2. No eval **answer** appears in train, for the free-text kinds (`qa`, `excerpt`, `abstain`).
   `subject` and `section` are exempt — their outputs are class labels and are *meant* to recur.

## Rebuilding

```bash
python scripts/generate_instructions.py           # rebuild train/eval
python scripts/generate_instructions.py --check   # verify committed files are current
python scripts/validate_data.py                   # full pre-flight
```

Generation is seeded (`SEED = 20260815`), so a rebuild is byte-identical unless you edit the
seeds or the corpus.

## Length budget

All records fit comfortably inside the 512-token limit notebook 02 truncates at — p50 ≈ 85
tokens, p95 ≈ 130, max ≈ 184 including instruction, input, and output. Nothing is being silently
truncated, which matters because a truncated target teaches the model to stop mid-sentence.
