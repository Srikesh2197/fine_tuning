# Domain corpus — provenance and design

## What this is

`raw/*.md` are **18 original documents written for this repository**. They describe AI/ML
engineering practice inside a large regulated bank: LLM platform architecture, retrieval,
evaluation, model risk governance, fine-tuning practice, monitoring, and third-party model risk.

`corpus.jsonl` is the built artifact — one JSON record per paragraph — produced by:

```bash
python scripts/build_corpus.py
```

Notebook 01 consumes `corpus.jsonl` and uses only the `text` field. The `doc`, `topic`, and
`section` fields exist for EDA and for `scripts/generate_instructions.py`, which uses document
structure to build classification tasks with a known answer.

## The institution is fictional

Every document describes **Meridian Trust**, an invented Tier-1 bank. Meridian Trust does not
exist. Nothing here is a real internal document from any institution, nothing was scraped, and
nothing is reproduced from a copyrighted source. The prose was written from scratch for this
repository.

This matters for two reasons. First, a fine-tuning demo built on real internal bank documents
would be a data-governance problem rather than a learning exercise. Second — see below — an
invented institution lets us plant facts that the base model provably cannot already know.

## The invented lexicon (and why it's there)

The documents share a consistent internal vocabulary. None of these are real terms:

| Term | What it denotes in the corpus |
|---|---|
| **Halton scale** | Model risk tiering, H1 (highest) through H4 (lowest) |
| **Gatepost** | Three-stage change-control gate before production |
| **Blue-file** | The mandatory model documentation artifact |
| **Sable** | The model inventory / system of record |
| **Lattice** | The internal LLM serving gateway |
| **Quarry** | The entitlements-aware retrieval service |
| **Scrim** | Sensitive-data detection and redaction |
| **Plumb** | The evaluation harness and release gate |
| **Ledgerline** | Dataset, feature, and lineage registry |
| **Amber window** | 30-day heightened-monitoring period after release |
| **DXI** | Drift index; 0.15 action threshold, 0.25 escalation |
| **40/40/20 rule** | Eval suite composition: golden / adversarial / production-sampled |

These give us **memorization probes**. `unsloth/Llama-3.2-1B` has never seen the sentence
"an H1 model is revalidated every six months," so if the fine-tuned model produces it and the
base model does not, the training demonstrably changed the weights. That is a far crisper signal
than a perplexity delta of a few tenths, and it is why notebook 01 probes these specific facts.

Some load-bearing numbers, repeated across documents so they are learnable:

- Revalidation cadence: **H1 = 6 months, H2 = 12, H3 = 24, H4 = on material change only**
- Amber window: **30 days**; review sampling **H1 100% / H2 25% / H3 5%**
- DXI thresholds: **0.15** action, **0.25** escalation
- Plumb gating: **2pp** golden regression, **3pp** production-sample regression
- Quarry defaults: **400-token chunks, 60-token overlap**; rerank top **50** to final **8**

## Statistics

| | |
|---|---|
| Documents | 18 |
| Paragraphs | 366 |
| Words | ~21,600 |
| Tokens | ~30,000 |
| 512-token training blocks | ~58 |

Honest note on scale: 30k tokens is a *small* corpus. It is enough to shift register, absorb
vocabulary, and memorize repeated facts — which is exactly what notebook 01 measures. It is not
enough to teach the model a field. Real domain-adaptive pretraining runs use 10<sup>8</sup>–10<sup>10</sup>
tokens. The tradeoff here is deliberate: the whole thing has to finish on a free Colab T4.

## Rebuilding

```bash
python scripts/build_corpus.py           # rebuild corpus.jsonl
python scripts/build_corpus.py --check   # verify the committed file is current
```

`build_corpus.py` drops heading lines, splits on blank lines, collapses soft-wrapped lines, and
filters paragraphs under 120 characters. Editing any `raw/*.md` file requires re-running the
build; `scripts/validate_data.py` fails if the committed artifact is stale.
