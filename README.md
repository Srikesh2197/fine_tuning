# Two-Stage LLM Fine-Tuning on a Free Colab T4

A hands-on walkthrough of the pipeline real fine-tuning programmes use: take a base model, adapt
it to a domain on raw text, then teach it to follow instructions. Both stages use LoRA, both run
end to end on a **free-tier Colab T4**, and both are instrumented so you can tell whether the
training actually did anything.

The domain is AI/ML engineering inside a large regulated bank.

```
unsloth/Llama-3.2-1B  (base — no instruction tuning)
      │
      │  notebook 01 — domain adaptation on ~30k tokens of raw text
      │                LoRA #1, ~11M params  ·  no instructions, no prompts
      ▼
   a model that writes fluent domain prose and cannot answer a question
      │
      │  merge LoRA #1 into the base weights
      │
      │  notebook 02 — instruction tuning on 295 (instruction, input, output) pairs
      │                LoRA #2, ~11M params  ·  loss on the response only
      ▼
   a model that answers in the domain's language — and stops
```

---

## Quick start

1. Push this repo to your own GitHub account.
2. Open notebook 01 in Colab, set the runtime to **T4 GPU**, and set `REPO_URL` in cell 2.
3. Run it top to bottom (~10 min). It saves a LoRA adapter to your Drive.
4. Open notebook 02 and run it top to bottom (~10 min). It picks the adapter up from Drive.

| | |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/Fine_tuning_demo/blob/main/notebooks/01_domain_adaptation_lora.ipynb) | **01 — Domain adaptation** (non-instructional, LoRA) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/Fine_tuning_demo/blob/main/notebooks/02_instruction_finetuning_lora.ipynb) | **02 — Instruction tuning** (LoRA on the merged stage-1 model) |

> Replace `YOUR-USERNAME` in the badge URLs and in each notebook's `REPO_URL`.
> The notebooks ship **without outputs** — run them yourself, the numbers are the point.

---

## What's in here

```
notebooks/
  01_domain_adaptation_lora.ipynb      non-instructional fine-tuning, baseline → train → measure
  02_instruction_finetuning_lora.ipynb merge stage 1, instruction-tune, three-way comparison
data/
  domain_corpus/raw/*.md               18 original documents (~21.6k words) — the training text
  domain_corpus/corpus.jsonl           built artifact: 366 paragraphs
  instruction/train.jsonl              295 synthetic instruction pairs
  instruction/eval.jsonl               34 held-out pairs
scripts/
  build_corpus.py                      raw/*.md → corpus.jsonl
  generate_instructions.py             corpus + hand-authored seeds → instruction pairs
  validate_data.py                     schema, freshness, dedup, leakage, token budget
docs/
  concepts.md                          the why: LoRA math, masking, evaluation, common bugs
```

Each data directory has its own README covering provenance and design.

---

## The corpus, and why it's fictional

The training text describes **Meridian Trust**, an invented Tier-1 bank. Nothing was scraped,
nothing is reproduced from a real institution's documents, and the prose was written from scratch
for this repo.

That isn't only about avoiding a data-governance problem. It gives us **memorization probes**.
`Llama-3.2-1B` has never encountered the *Halton scale*, *Gatepost*, *Blue-file*, or *DXI*, so if
the fine-tuned model completes "an H1 model must be revalidated every" with "six months", that
string demonstrably came from our training data and nowhere else.

That's a much crisper signal than a perplexity delta of a few tenths — which is exactly why the
corpus was designed this way. See [`data/domain_corpus/README.md`](data/domain_corpus/README.md).

---

## What this does differently

Built with [sunnysavita10/Complete-LLM-Finetuning](https://github.com/sunnysavita10/Complete-LLM-Finetuning)
(folders 14 and 15) as a reference. The arc is theirs and it's the right arc. These four things
are done differently, because the reference implementation has defects that make a pipeline look
like it works when it doesn't:

| | Common approach | Here |
|---|---|---|
| **Loading the adapter** | `AutoModelForCausalLM.from_pretrained(lora_dir)` — silently returns the **base model**, ignoring the adapter files next to it | `PeftModel.from_pretrained` + `merge_and_unload()`, with an assertion that the weights actually changed |
| **Tokenization** | `padding="max_length"` + `labels = input_ids.copy()` — trains the model to predict pad tokens | Stage 1 packs into fixed blocks (100% real tokens); stage 2 pads dynamically with `labels = -100` |
| **Instruction loss** | Supervises the whole sequence — the model learns to generate `### Instruction:` headers | Completion-only masking, with a decoded sanity check showing the boundary |
| **Evidence** | No EOS on targets, no eval split, no baseline | EOS appended and generation length measured; held-out splits; perplexity captured **before** training |

`docs/concepts.md` §11 documents each one, so the lesson survives beyond this repo.

Two other choices worth flagging:

- **No quantization.** The 1B model loads in fp16, not 4-bit. Not because QLoRA is bad — because
  you cannot cleanly `merge_and_unload()` into quantized weights, and the merge between stages is
  the whole architecture here.
- **No TRL.** Plain `Trainer` + `peft`, so the masking and packing are visible rather than
  handled by `SFTTrainer` behind the scenes. Use TRL in production; write it by hand once first.

---

## Working locally

The scripts are pure standard library — no GPU, no ML dependencies:

```bash
python scripts/build_corpus.py            # rebuild corpus.jsonl from raw/*.md
python scripts/generate_instructions.py   # rebuild train/eval instruction data
python scripts/validate_data.py           # full pre-flight: schema, freshness, leakage, budgets
```

Generation is seeded, so rebuilds are byte-identical unless you change the inputs.
`validate_data.py` fails if a committed artifact is stale.

Editing the corpus is the intended way to make this yours: drop your own documents into
`data/domain_corpus/raw/`, add an entry to `DOC_SUBJECT` in `build_corpus.py`, rebuild, and rerun.

---

## Results

Fill these in when you run it — the numbers depend on your Colab session.

| Metric | Baseline | After stage 1 |
|---|---|---|
| Perplexity, held-out domain text | | |
| Perplexity, out-of-domain text | | |
| Reproduces the invented facts | no | ? |

| Question | base | + domain | + domain + instruct |
|---|---|---|---|
| "How often must an H1 model be revalidated?" | | | |

Roughly what to expect: a real but **modest** perplexity improvement (30k tokens is a small
corpus), clear adoption of the corpus vocabulary, visible overfitting in the last epochs of stage
1, and an unmistakable behavioural change after stage 2 — the model answers and stops, where
before it continued indefinitely.

---

## Environment

Pinned in [`requirements-colab.txt`](requirements-colab.txt), verified against PyPI and the
transformers v5 source:

```
transformers==5.15.0    peft==0.20.0    datasets==5.0.1    accelerate==1.14.0
```

`torch` is deliberately unpinned — Colab ships a CUDA-matched build and replacing it is slow and
fragile. A known-good fallback set is commented in the same file, for if the pins ever conflict.

**Base model:** [`unsloth/Llama-3.2-1B`](https://huggingface.co/unsloth/Llama-3.2-1B) — an ungated
mirror of Meta's Llama 3.2 1B **base** weights. Same weights as
[`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B), without the licence
gate and HF token setup. Use the official repo instead if you'd rather; only `MODEL_ID` changes.

Note it's the **base** model, not `-Instruct`. That's the point: the whole narrative is watching
instruction-following get installed.

---

## Credit

Structure and sequencing follow [sunnysavita10/Complete-LLM-Finetuning](https://github.com/sunnysavita10/Complete-LLM-Finetuning).
The corpus, instruction data, notebooks, and implementation here are original.
