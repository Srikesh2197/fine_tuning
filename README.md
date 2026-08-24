# Two-Stage LLM Fine-Tuning on a Free Colab T4

A hands-on walkthrough of the pipeline real fine-tuning programmes use: take a base model, adapt
it to a domain on raw text, then teach it to follow instructions. Both stages use LoRA, both run
end to end on a **free-tier Colab T4**, and both are instrumented so you can tell whether the
training actually did anything.

Everything comes from one real, public dataset:
**[`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA)** (MIT licence).

```
unsloth/Llama-3.2-1B  (base — no instruction tuning)
      │
      │  notebook 01 — domain adaptation on ~292k tokens of raw PubMed abstracts
      │                LoRA #1, ~11M params  ·  no instructions, no prompts
      ▼
   a model that writes fluent abstract prose and cannot answer a question
      │
      │  merge LoRA #1 into the base weights
      │
      │  notebook 02 — instruction tuning on 800 expert-annotated Q/A pairs
      │                LoRA #2, ~11M params  ·  loss on the response only
      ▼
   a model that answers yes/no/maybe with a justification — and stops
```

The payoff is a number, not a vibe: stage 2 is scored on **decision accuracy over 200 held-out
articles**, against a **55.0% majority-class baseline** and against the untuned base model.

---

## Quick start

1. Open notebook 01 in Colab (badge below) and set the runtime to **T4 GPU**
   (Runtime -> Change runtime type -> T4 GPU).
2. Run it top to bottom (~12 min). It saves a LoRA adapter to your Google Drive.
3. Open notebook 02 and run it top to bottom (~12 min). It picks the adapter up from Drive.

| | |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Srikesh2197/fine_tuning/blob/main/notebooks/01_domain_adaptation_lora.ipynb) | **01 - Domain adaptation** (non-instructional, LoRA) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Srikesh2197/fine_tuning/blob/main/notebooks/02_instruction_finetuning_lora.ipynb) | **02 - Instruction tuning** (LoRA on the merged stage-1 model) |

> Badges open the notebooks from `main`. The repo is private, so the first time you use
> one, Colab will ask to authorise GitHub access — tick **Include private repos**.
> The notebooks ship **without outputs** - run them yourself, the numbers are the point.

**The notebooks are self-contained.** They pull data from Hugging Face with `load_dataset()` and
need no repo checkout, no clone, and no Drive copy of the repo — so the private-repo problem goes
away entirely. Drive is still used for one thing: handing the stage-1 adapter to notebook 02.

---

## What's in here

```
notebooks/
  01_domain_adaptation_lora.ipynb      non-instructional fine-tuning, baseline → train → measure
  02_instruction_finetuning_lora.ipynb merge stage 1, instruction-tune, score against baselines
scripts/
  prepare_data.py                      materialise the prepared data locally (no GPU needed)
  validate_data.py                     schema, disjointness, stratification, token budgets
data/
  README.md                            dataset provenance, splits, licence, honest caveats
docs/
  concepts.md                          the why: LoRA math, masking, evaluation, common bugs
```

Nothing is committed under `data/` — the notebooks load from the Hub at runtime.
`scripts/prepare_data.py` writes JSONL there for local inspection (gitignored).

---

## The data, and why this dataset

The requirement was one authentic source feeding **both** stages. PubMedQA does, via configs built
from different PubMed articles:

| stage | config | what we use |
|---|---|---|
| 01 — domain adaptation | `pqa_unlabeled` (61,249) | 1,000 abstracts, **questions discarded** — ~292k tokens of raw prose |
| 02 — instruction tuning | `pqa_labeled` (1,000) | question + abstract → `Answer: yes/no/maybe` + expert justification |

Three properties make it a good teaching set:

1. **Genuinely specialised text.** A 1B general model is measurably worse at biomedical abstracts
   than at Wikipedia, so stage 1 has real headroom. Fine-tuning on SQuAD or Dolly contexts — both
   Wikipedia — would move perplexity almost not at all.
2. **Stage 2 is gradable.** Every record carries an expert yes/no/maybe, so accuracy is a number.
3. **The stages are disjoint at article level**, so stage 2's score isn't measuring what stage 1
   memorised. Both notebooks assert this rather than trusting it.

`data/README.md` has the full breakdown. `docs/concepts.md` §12 covers what else was considered
and the criteria to apply to your own data.

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
| **Instruction loss** | Supervises the whole sequence — here that would spend **88%** of the budget regurgitating abstracts | Completion-only masking, with a decoded sanity check showing the boundary |
| **Evidence** | No EOS on targets, no eval split, no baseline | EOS appended and generation length measured; disjoint splits; perplexity captured **before** training; accuracy against a majority-class baseline |

`docs/concepts.md` §11 documents each one, so the lesson survives beyond this repo.

Two other choices worth flagging:

- **No quantization.** The 1B model loads in fp16, not 4-bit. Not because QLoRA is bad — because
  you cannot cleanly `merge_and_unload()` into quantized weights, and the merge between stages is
  the whole architecture here.
- **No TRL.** Plain `Trainer` + `peft`, so the masking and packing are visible rather than
  handled by `SFTTrainer` behind the scenes. Use TRL in production; write it by hand once first.

---

## Working locally

No GPU needed to inspect the data:

```bash
pip install datasets transformers
python scripts/prepare_data.py        # writes data/*.jsonl for inspection (gitignored)
python scripts/validate_data.py       # full pre-flight
```

`validate_data.py` checks schema, article-level disjointness between stages, split stratification,
that the yes/no/maybe grading regex recovers the gold label on 100% of records, and that nothing
exceeds the notebooks' token budgets. It also parses both notebooks and fails if the constants
they inline have drifted from `prepare_data.py`.

To point this at your own data, replace the two `load_dataset` blocks. `docs/concepts.md` §12 has
the criteria for picking a replacement.

---

## Results

Fill these in when you run it — the numbers depend on your Colab session.

**Stage 1 — domain adaptation**

| Metric | Baseline | After stage 1 |
|---|---|---|
| Perplexity, held-out PubMed abstracts | | |
| Perplexity, out-of-domain prose | | |
| 8-gram overlap with training continuations | n/a | |

**Stage 2 — decision accuracy on 200 held-out articles**

| System | Accuracy | Parse rate |
|---|---|---|
| Majority class (always "yes") | 55.0% | n/a |
| base | | |
| + domain | | |
| + domain + instruct | | |

What to expect: a clear perplexity drop in stage 1 (biomedical text has real headroom for a 1B
general model), near-zero parse rates for the two untuned systems — neither has ever been taught
to emit `Answer:` — and a stage-2 model that answers in the right format and stops. Whether it
beats 55% on *content* is the genuinely open question at this data scale; the notebook prints a
confusion matrix so you can see whether it's discriminating or just saying "yes".

---

## Environment

Pinned in [`requirements-colab.txt`](requirements-colab.txt), verified against PyPI and the
transformers v5 source:

```
transformers==5.15.0    peft==0.20.0    datasets==5.0.1    accelerate==1.14.0
```

`torch` is deliberately unpinned — Colab ships a CUDA-matched build and replacing it is slow and
fragile. Colab's preinstalled `torchao` is uninstalled, because `peft`'s optional integration
raises rather than degrading when it finds a version below its minimum. A known-good fallback pin
set is commented in the same file.

**Base model:** [`unsloth/Llama-3.2-1B`](https://huggingface.co/unsloth/Llama-3.2-1B) — an ungated
mirror of Meta's Llama 3.2 1B **base** weights. Same weights as
[`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B), without the licence
gate and HF token setup. Only `MODEL_ID` changes if you'd rather use the official repo.

Note it's the **base** model, not `-Instruct`. That's the point: the whole narrative is watching
instruction-following get installed.

---

## Caveats

- **This is not medical software.** A 1B model fine-tuned on 800 abstracts is a training-mechanics
  exercise. Nothing it produces is medical advice.
- **Our split is not the official PubMedQA benchmark split** (450/50/500). Don't compare these
  numbers to published leaderboard results.
- **Both stages are data-starved by design**, to fit a free Colab session. `docs/concepts.md` §13
  lists what to scale first.

## Credit

Structure and sequencing follow [sunnysavita10/Complete-LLM-Finetuning](https://github.com/sunnysavita10/Complete-LLM-Finetuning).
Data is [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (Jin et al., EMNLP 2019,
[paper](https://arxiv.org/abs/1909.06146)). Notebooks and implementation here are original.
