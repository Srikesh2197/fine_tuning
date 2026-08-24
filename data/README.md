# Data — PubMedQA

Both fine-tuning stages come from one real, public dataset:

**[`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA)** · MIT licence ·
[paper](https://arxiv.org/abs/1909.06146) (Jin et al., EMNLP 2019)

Nothing is committed here. The notebooks call `load_dataset()` at runtime, so they are
self-contained and need no repo checkout. This directory holds the provenance notes, and
`scripts/prepare_data.py` will materialise the prepared JSONL locally if you want to inspect the
data without a GPU (output is gitignored).

## Why this dataset

The requirement was one authentic source that feeds **both** stages. PubMedQA does, because it
ships three configs built from different PubMed articles:

| config | rows | used for |
|---|---|---|
| `pqa_unlabeled` | 61,249 | **Stage 1** — raw abstract prose, questions discarded |
| `pqa_labeled` | 1,000 | **Stage 2** — expert-annotated question / abstract / yes-no-maybe / justification |
| `pqa_artificial` | 211,269 | unused here; auto-labelled, a good way to scale stage 2 |

Three properties make it a better teaching set than the alternatives:

1. **Genuinely specialised text.** A 1B general model is measurably worse at biomedical abstracts
   than at Wikipedia, so stage 1 has real headroom. Fine-tuning on SQuAD or Dolly contexts — both
   Wikipedia — would move perplexity almost not at all, because the model already knows them.
2. **Stage 2 has an objective metric.** Every labelled record carries a `final_decision` of
   yes/no/maybe, so accuracy is measurable rather than a matter of squinting at generated text.
3. **The stages are disjoint at article level.** `pqa_unlabeled` and `pqa_labeled` cover different
   papers, so stage 2's metric isn't measuring what stage 1 memorised. Both the notebooks and
   `validate_data.py` assert this rather than trusting it.

## A raw record

```python
{"pubid": 21645374,
 "question": "Do mitochondria play a role in remodelling lace plant leaves during PCD?",
 "context": {"contexts": ["Programmed cell death (PCD) is the regulated death of cells ...", ...],
             "labels":   ["BACKGROUND", "RESULTS", ...]},
 "long_answer": "Results depicted mitochondrial dynamics in vivo as PCD progresses ...",
 "final_decision": "yes"}
```

## Stage 1 — domain adaptation (notebook 01)

Sample `N_ABSTRACTS = 1000` articles from `pqa_unlabeled` (seeded), keep only
`context.contexts`, and **throw the questions away entirely**. That is what makes it
non-instructional: the training signal is plain next-token prediction over prose.

Sections shorter than 120 characters are dropped as fragments.

| | |
|---|---|
| Articles | 1,000 |
| Abstract sections | 2,989 |
| Words | ~196,000 |
| Tokens (Llama 3.2) | **~292,600** |
| 512-token training blocks | ~571 |

Biomedical text tokenizes at **~1.50 tokens/word** against ~1.3 for general English — technical
terms fragment into more subwords. That inflates every downstream budget and is worth knowing.

## Stage 2 — instruction tuning (notebook 02)

All 1,000 `pqa_labeled` records become Alpaca-style triples:

- **instruction** — the task, plus the research question
- **input** — the abstract, rendered as `LABEL: text` blocks
- **output** — `Answer: <decision>` then the expert's justification

Putting the decision on the first line is deliberate: it makes the answer machine-gradable with a
regex, and it teaches the model to commit before it explains.

Split 800 / 200, **stratified by decision** — `maybe` is only 11% of the data, and an unstratified
split would leave the eval set with an unstable number of them.

| | train | eval |
|---|---|---|
| yes | 442 | 110 |
| no | 270 | 68 |
| maybe | 88 | 22 |
| **total** | **800** | **200** |

**Majority-class baseline: 55.0%** (always answer "yes"). Any model that can't beat that has
learned nothing useful, and it's the number the notebook reports alongside every result.

Sequence lengths: p50 451, p90 571, max 878 tokens. `MAX_LEN = 896` truncates nothing — which
matters, because a truncated target teaches the model to stop mid-sentence.

## Not the official benchmark split

The official PubMedQA split is 450 train / 50 dev / **500 test**. Ours is 800/200 so that training
and evaluation both fit a free Colab session. **Do not compare these numbers to published
leaderboard results.** For reference, fine-tuned 7B–70B models sit around 55–78% on the official
test split, so a 1B model on 800 examples beating 55% at all would be a decent showing.

## Reproducing locally

```bash
pip install datasets transformers
python scripts/prepare_data.py       # writes data/*.jsonl (gitignored)
python scripts/validate_data.py      # schema, disjointness, stratification, token budgets
```

`prepare_data.py` and the notebooks share the same constants (`SEED`, `N_ABSTRACTS`,
`MIN_SECTION_CHARS`, `EVAL_FRACTION`, `TASK`). Because the notebooks inline the logic to stay
self-contained, those constants exist in two places — `validate_data.py` parses the notebooks and
fails if they drift apart.

## Licence and use

PubMedQA is MIT-licensed. The underlying abstracts are from PubMed; this repo redistributes none
of them, it just loads the dataset from the Hub at runtime.

**Nothing here is medical advice.** A 1B model fine-tuned on 800 abstracts is a training-mechanics
exercise, not a clinical tool.
