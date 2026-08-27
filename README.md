# Three-Stage LLM Fine-Tuning on a Free Colab T4

A hands-on walkthrough of the pipeline real fine-tuning programmes use: take a base model, adapt
it to a domain on raw text, teach it to follow instructions, then tune it on preferences. All three
stages use LoRA, all three run end to end on a **free-tier Colab T4**, and all three are
instrumented so you can tell whether the training actually did anything.

Everything comes from one real, public dataset:
**[`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA)** (MIT licence).

A fourth notebook runs alongside the pipeline rather than after it: **QLoRA at 8B**, 4-bit base
weights, scored on the same held-out records as stage 2. That one needs an A100 or L4 — see
[Notebook 04](#notebook-04--qlora-at-8b) below.

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
      │
      │  merge LoRA #2 into the base weights
      │
      │  notebook 03 — preference tuning with DPO, on pairs the model generates itself
      │                LoRA #3, ~11M params  ·  loss on the margin between two answers
      ▼
   a model tuned to rank its own good answers above its own bad ones
```

The payoff is a number, not a vibe: stages 2 and 3 are scored on **decision accuracy over the same
200 held-out articles**, against a **55.0% majority-class baseline** and against the untuned base
model. Stage 3 adds a **paired significance test**, because at n=200 an accuracy delta under about
7 points is not distinguishable from noise — and preference tuning is the stage most likely to
produce one.

---

## Quick start

1. Open notebook 01 in Colab (badge below) and set the runtime to **T4 GPU**
   (Runtime -> Change runtime type -> T4 GPU).
2. Run it top to bottom (~12 min). It saves a LoRA adapter to your Google Drive.
3. Open notebook 02 and run it top to bottom (~12 min). It picks the adapter up from Drive.
4. Open notebook 03 and run it top to bottom (~35 min). It needs **both** earlier adapters.
5. Optional, and **not free-tier**: notebook 04 (QLoRA at 8B, ~60 min) needs an **A100 or L4**.
   It is standalone — no adapters from the other three, nothing handed to them.

| | |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Srikesh2197/fine_tuning/blob/main/notebooks/01_domain_adaptation_lora.ipynb) | **01 - Domain adaptation** (non-instructional, LoRA) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Srikesh2197/fine_tuning/blob/main/notebooks/02_instruction_finetuning_lora.ipynb) | **02 - Instruction tuning** (LoRA on the merged stage-1 model) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Srikesh2197/fine_tuning/blob/main/notebooks/03_preference_tuning_dpo.ipynb) | **03 - Preference tuning** (DPO on self-generated pairs) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Srikesh2197/fine_tuning/blob/main/notebooks/04_qlora_instruction_tuning.ipynb) | **04 - QLoRA at 8B** (4-bit NF4, standalone) — **A100/L4, not T4** |

> Badges open the notebooks from `main`. The repo is private, so the first time you use
> one, Colab will ask to authorise GitHub access — tick **Include private repos**.
> The notebooks ship **without outputs** - run them yourself, the numbers are the point.

**The notebooks are self-contained.** They pull data from Hugging Face with `load_dataset()` and
need no repo checkout, no clone, and no Drive copy of the repo — so the private-repo problem goes
away entirely. Drive is still used only for handing artifacts between stages: the stage-1 adapter
to notebook 02, both adapters to notebook 03, and a cache of notebook 03's mined preference pairs
so a re-run skips its slowest cell.

---

## What's in here

```
notebooks/
  01_domain_adaptation_lora.ipynb      non-instructional fine-tuning, baseline → train → measure
  02_instruction_finetuning_lora.ipynb merge stage 1, instruction-tune, score against baselines
  03_preference_tuning_dpo.ipynb       merge stages 1+2, mine preference pairs, DPO, re-score
  04_qlora_instruction_tuning.ipynb    4-bit NF4 on an 8B base, and what merging into it costs
scripts/
  prepare_data.py                      materialise the prepared data locally (no GPU needed)
  validate_data.py                     schema, disjointness, stratification, token budgets
data/
  README.md                            dataset provenance, splits, licence, honest caveats
docs/
  concepts.md                          the why: LoRA math, masking, DPO, evaluation, common bugs
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
| 03 — preference tuning | `pqa_labeled`, the same 800 training rows | (chosen, rejected) pairs **mined from the stage-2 model's own samples**, graded against the expert label |
| 04 — QLoRA (parallel) | `pqa_labeled` | the **same** 800 train / 200 eval split as stage 2, so the 8B number is comparable |

Three properties make it a good teaching set:

1. **Genuinely specialised text.** A 1B general model is measurably worse at biomedical abstracts
   than at Wikipedia, so stage 1 has real headroom. Fine-tuning on SQuAD or Dolly contexts — both
   Wikipedia — would move perplexity almost not at all.
2. **Stage 2 is gradable.** Every record carries an expert yes/no/maybe, so accuracy is a number.
3. **The stages are disjoint at article level**, so stage 2's score isn't measuring what stage 1
   memorised. Every notebook asserts this rather than trusting it.

Stage 3 needs no new data. Its preference pairs are generated by sampling the stage-2 model twice
per training question and grading both answers with the same regex — so the negatives come from the
policy actually being trained, which is what DPO assumes and hand-written pairs are not.

`data/README.md` has the full breakdown. `docs/concepts.md` §14 covers what else was considered
and the criteria to apply to your own data.

---

## What this does differently

Built with [sunnysavita10/Complete-LLM-Finetuning](https://github.com/sunnysavita10/Complete-LLM-Finetuning)
(folders 14, 15 and 16) as a reference. The arc is theirs and it's the right arc. These five things
are done differently, because the reference implementation has defects that make a pipeline look
like it works when it doesn't:

| | Common approach | Here |
|---|---|---|
| **Loading the adapter** | `AutoModelForCausalLM.from_pretrained(lora_dir)` — silently returns the **base model**, ignoring the adapter files next to it | `PeftModel.from_pretrained` + `merge_and_unload()`, with an assertion that the weights actually changed |
| **Tokenization** | `padding="max_length"` + `labels = input_ids.copy()` — trains the model to predict pad tokens | Stage 1 packs into fixed blocks (100% real tokens); stage 2 pads dynamically with `labels = -100` |
| **Instruction loss** | Supervises the whole sequence — here that would spend **88%** of the budget regurgitating abstracts | Completion-only masking, with a decoded sanity check showing the boundary |
| **Evidence** | No EOS on targets, no eval split, no baseline | EOS appended and generation length measured; disjoint splits; perplexity captured **before** training; accuracy against a majority-class baseline |
| **Preference data** | 5 hand-written pairs in a CSV, all five used for training, and `load_in_8bit=True` immediately before `merge_and_unload()` | A few hundred pairs mined from the stage-2 model's own sampled mistakes, with a length-bias audit, a leakage assert, a held-out pair split, and a paired significance test |

`docs/concepts.md` §13 documents each one, so the lesson survives beyond this repo.

Two other choices worth flagging:

- **No quantization in stages 1-3.** The 1B model loads in fp16, not 4-bit. Not because QLoRA is
  bad — because the merge between stages is the whole architecture here, and merging into
  quantized weights is lossy. Notebook 04 is where QLoRA lives, structured so that it never has to
  merge, and section 14 of it puts an actual number on what the merge would have cost.
- **TRL only where it earns its place.** Stages 1 and 2 use plain `Trainer` + `peft`, so the
  masking and packing stay visible rather than handled by `SFTTrainer` behind the scenes — write it
  by hand once first. Stage 3 uses TRL's `DPOTrainer`, because the DPO loss is subtle and TRL's is
  the reference implementation. The pairs, the reference policy and every metric around it are
  still built and checked by hand.

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
exceeds the notebooks' token budgets. It also parses all four notebooks and fails if the constants
they inline have drifted from `prepare_data.py`.

To point this at your own data, replace the two `load_dataset` blocks. `docs/concepts.md` §14 has
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

**Stages 2 and 3 — decision accuracy on the same 200 held-out articles**

| System | Accuracy | Parse rate | Mean tokens |
|---|---|---|---|
| Majority class (always "yes") | 55.0% | n/a | n/a |
| base | | | |
| + domain | | | |
| + domain + instruct | | | |
| + domain + instruct + dpo | | | |

**Notebook 04 — QLoRA at 8B, on the same 200 held-out articles**

| System | Accuracy | Parse rate | Mean tokens |
|---|---|---|---|
| Majority class (always "yes") | 55.0% | n/a | n/a |
| 8B 4-bit base (adapter off) | | | |
| 8B 4-bit + QLoRA | | | |
| 8B 4-bit + QLoRA, merged in | | | |

| Quantization measurement | Value |
|---|---|
| 4-bit footprint / fp16 equivalent | / 16.1 GB |
| Footprint after `prepare_model_for_kbit_training` | |
| Peak GPU memory during training | |
| Requantization error / LoRA update, `‖err‖ / ‖dW‖` | |
| McNemar p, merged vs unmerged | |

**Stage 3 — the preference objective itself**

| Metric | Step 0 | Final |
|---|---|---|
| Reward margin, held-out pairs | 0.0000 | |
| `rewards/chosen` | 0.0000 | |
| Preference accuracy, held-out pairs | | |
| McNemar p, stage 3 vs stage 2 | n/a | |

What to expect: a clear perplexity drop in stage 1 (biomedical text has real headroom for a 1B
general model), near-zero parse rates for the two untuned systems — neither has ever been taught
to emit `Answer:` — and a stage-2 model that answers in the right format and stops. Whether it
beats 55% on *content* is the genuinely open question at this data scale; the notebook prints a
confusion matrix so you can see whether it's discriminating or just saying "yes".

For notebook 04, expect the 4-bit base to parse near 0% (it has never been taught to emit
`Answer:`) and the QLoRA row to answer in the right format. Whether 8B beats the 1B pipeline on
*content* is confounded — that pipeline had a domain-adaptation stage this notebook does not — so
read it as scale context, not as a controlled result. The measurement worth trusting is
`‖err‖ / ‖dW‖`: it says how much of the trained update a merge into 4-bit would throw away.

For stage 3, expect the reward margin to rise and `rewards/chosen` to go **negative** — DPO widens
the margin mostly by suppressing the rejected answer rather than promoting the chosen one. That is
normal, and it is also why the margin alone proves nothing. The accuracy row and the McNemar
p-value are what decide whether anything real happened, and "no detectable change" is a legitimate
and fairly likely outcome at a few hundred pairs.

---

## Environment

Pinned in [`requirements-colab.txt`](requirements-colab.txt), verified against PyPI and the
transformers v5 source:

```
transformers==5.15.0    peft==0.20.0    datasets==5.0.1    accelerate==1.14.0
trl==0.29.1             bitsandbytes==0.50.2
```

`trl` is used by notebook 03 only. Its `DPOConfig` **moves several defaults** — `learning_rate` to
`1e-6`, `gradient_checkpointing` to `True`, `bf16` to auto-on (which a Turing T4 then emulates), and
`loss_type` to a `list` rather than a string — and each one fails quietly rather than loudly.
Notebook 03 sets all of them explicitly, and `requirements-colab.txt` records which layer moves
which.

`bitsandbytes` is used by notebook 04 only. Two of `BitsAndBytesConfig`'s defaults are worth
overriding rather than inheriting: `bnb_4bit_compute_dtype` defaults to **fp32** — it is the matmul
precision, not the storage precision, so the default makes you pay for precision the 4-bit weights
cannot supply — and `bnb_4bit_quant_type` defaults to `"fp4"` when `"nf4"` is strictly better at
the same bit cost. `requirements-colab.txt` records both, the same way it records what `DPOConfig`
moves.

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

### Notebook 04 — QLoRA at 8B

Notebooks 01-03 run on a free T4. **Notebook 04 does not**, and asserts an Ampere-or-later GPU
rather than falling back: 4-bit compute on a GPU with no native bfloat16 turns a one-hour run into
most of an afternoon. It defaults to
[`unsloth/Meta-Llama-3.1-8B`](https://huggingface.co/unsloth/Meta-Llama-3.1-8B) (ungated, and it
shares a tokenizer with the 1B, so notebook 02's token budget carries over unchanged), with
[`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B) one switch away.

It is standalone: it takes no adapter from the earlier notebooks and hands none back. That is
deliberate — stages 1→2→3 merge between stages, and merging into 4-bit weights is the operation
this repo argues against. Training a single adapter means never having to.

---

## Caveats

- **This is not medical software.** A 1B model fine-tuned on 800 abstracts is a training-mechanics
  exercise. Nothing it produces is medical advice.
- **Our split is not the official PubMedQA benchmark split** (450/50/500). Don't compare these
  numbers to published leaderboard results.
- **All three stages are data-starved by design**, to fit a free Colab session.
  `docs/concepts.md` §15 lists what to scale first.
- **A 200-record accuracy difference under ~7 points is noise.** That is roughly the 95% interval
  for a proportion near 55% at n=200. Notebook 03 reports a paired McNemar test precisely so that a
  stage-3 result inside that band gets called what it is.
- **Notebook 04 is not free-tier**, and its 8B-vs-1B comparison is confounded by the missing
  domain-adaptation stage as well as by scale. It is a QLoRA demonstration with an honest metric
  attached, not a clean scaling study.
- **Stage 3 tunes a proxy for preference, not preference.** "Chosen" means "agreed with the
  expert's yes/no/maybe". Real preference data encodes helpfulness, hedging and tone, none of which
  a regex can grade.

## Credit

Structure and sequencing follow [sunnysavita10/Complete-LLM-Finetuning](https://github.com/sunnysavita10/Complete-LLM-Finetuning)
(folders 14, 15 and 16).
Data is [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (Jin et al., EMNLP 2019,
[paper](https://arxiv.org/abs/1909.06146)). Notebooks and implementation here are original.
