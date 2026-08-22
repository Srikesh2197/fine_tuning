# Concepts — the reasoning behind every decision in this repo

Notes written while building this. The notebooks show *what*; this explains *why*, and records
the mistakes that are easy to make and hard to notice.

---

## 1. Where fine-tuning sits

```
pretraining          trillions of tokens of raw text, next-token prediction
     │               → a model that continues text
     ▼
domain adaptation    10⁶–10¹⁰ tokens of raw DOMAIN text, same objective        ← notebook 01
     │               → a model that continues text in your domain's register
     ▼
instruction tuning   10³–10⁶ (request, response) pairs, loss on response only  ← notebook 02
     │               → a model that answers when addressed
     ▼
preference tuning    (chosen, rejected) pairs — DPO / ORPO / RLHF              ← not covered
                     → a model that answers the way people prefer
```

Every stage optimises the **same** objective: predict the next token. What changes is the data
and which tokens are scored. That's the whole idea, and holding onto it makes the rest obvious.

### Non-instructional fine-tuning, specifically

The term is confusing because it names a stage by what it lacks. Concretely:

|                  | Domain adaptation | Instruction tuning |
|---|---|---|
| Data | Raw text | `(instruction, input, output)` triples |
| Structure | None — just prose | A template with delimiters |
| Loss over | Every token | Response tokens only |
| Teaches | Vocabulary, register, facts | Behaviour: answer, stop, decline |
| Evaluate with | Perplexity on held-out domain text | Generation quality, held-out instructions |

**A domain-adapted model still won't answer questions.** This surprises people constantly. They
train on their documents, ask "what is X?", get back a stream of further questions, and conclude
the training failed. It didn't — it did exactly what raw text asks for. Judging stage 1 by
instruction-following is a category error.

### Order matters

Domain adaptation **first**, instruction tuning **second**. The reverse damages the
instruction-following the second stage installs, because training on raw text pulls the model
back toward continuation.

---

## 2. LoRA

Freeze the pretrained weight $W \in \mathbb{R}^{d \times k}$ and learn a low-rank update beside it:

$$W' = W + \Delta W = W + \frac{\alpha}{r}BA, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k}$$

Only $A$ and $B$ train. $A$ is initialised randomly and $B$ to zero, so $\Delta W = 0$ at step
zero — training starts from exactly the pretrained model.

**Concretely, for `Llama-3.2-1B` at $r=16$ on all seven projections:**

| module | shape | LoRA params |
|---|---|---|
| `q_proj` | 2048 → 2048 | 16 × (2048+2048) = 65,536 |
| `k_proj` | 2048 → 512 | 40,960 |
| `v_proj` | 2048 → 512 | 40,960 |
| `o_proj` | 2048 → 2048 | 65,536 |
| `gate_proj` | 2048 → 8192 | 163,840 |
| `up_proj` | 2048 → 8192 | 163,840 |
| `down_proj` | 8192 → 2048 | 163,840 |
| | per layer | **704,512** |
| | × 16 layers | **≈ 11.3M** |

11.3M / 1.24B ≈ **0.9% trainable**. The adapter is ~45 MB on disk against ~2.5 GB for the model.

### `r` and `lora_alpha`

`r` is capacity. `alpha/r` is the scaling on the update — how hard it pushes. The convention
`alpha = 2r` means raising `r` adds capacity without changing effective step size.

| purpose | rank |
|---|---|
| Format, register, tone | 8 |
| Domain adaptation, new vocabulary | 16–32 |
| Larger — rarely justified on small corpora | 64+ |

### `target_modules` matters more than `r`

The `["q_proj", "v_proj"]` default comes from the original LoRA paper, which studied
*behavioural* adaptation. It is a poor default for domain adaptation.

Attention decides **what to look at**; the feed-forward layers are where **factual and lexical
association** is stored. If you want the model to absorb a vocabulary — `Halton tier`, `Gatepost`,
`DXI` — the MLP is where that lands. Both notebooks target all seven projections.

If domain adaptation underdelivers, check `target_modules` before touching `r`.

### Why adapters rather than full fine-tuning

Practical, not ideological:

1. Trains on hardware you have.
2. ~45 MB artifacts you can version, diff, and swap at serving time.
3. Base weights are untouched, so `disable_adapter()` gives you an **exact** A/B against the
   original — no second model in memory. Notebook 01 uses this for the before/after comparison and
   notebook 02 for the three-way table.
4. Rollback is repointing a path, not redeploying 2.5 GB.

Point 3 is underrated. Cheap counterfactuals are what make "did this training actually do
anything?" a question you can answer in one line.

---

## 3. Why merge between stages

Notebook 02 does base → attach stage-1 adapter → `merge_and_unload()` → attach a *fresh* stage-2
adapter. It does **not** stack two adapters.

Merging folds $\frac{\alpha}{r}BA$ into $W$, giving a plain `LlamaForCausalLM` with the domain
knowledge baked in and no PEFT wrapper. Stage 2 then starts from a clean base.

Stacking two live adapters instead means their updates interact at every forward pass, in ways
that are hard to predict and harder to attribute when something regresses. Merging is also how
you'd hand stage 1 to someone else as a base model.

**One consequence worth knowing:** you cannot cleanly merge into quantized weights. That is why
this repo loads the 1B model in fp16 rather than using QLoRA — the merge in notebook 02 is the
whole architecture, and 4-bit would break it. At 7B+ you'd need QLoRA and would have to
restructure (merge in fp16 on CPU, or keep the adapters stacked and accept the coupling).

---

## 4. Packing vs padding

**The common tutorial pattern:**

```python
tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512)
tokens["labels"] = tokens["input_ids"].copy()      # ← the bug
```

Two things go wrong:

1. Most of each sequence is padding. In this repo's corpus, ~84% of token slots.
2. Because labels are a straight copy, **pad positions are supervised**. The model is explicitly
   trained to predict `<pad>` after `<pad>`.

The loss curve looks *great*, because predicting padding is trivially easy and padding dominates
the average. Nothing warns you.

**The fix, depending on the stage:**

- **Stage 1 (raw text):** concatenate everything into one stream and slice into fixed 512-token
  blocks. No padding at all, 100% of supervised tokens real. This is how pretraining works, and
  continued pretraining should match it.
- **Stage 2 (variable-length pairs):** pad dynamically per batch, and pad `labels` with `-100`.

`-100` is PyTorch's `ignore_index` for `CrossEntropyLoss`: those positions contribute nothing.

Notebook 01 measures both approaches side by side rather than asserting the difference.

---

## 5. Completion-only loss masking

The single highest-leverage detail in instruction tuning.

A training example renders as:

```
Below is an instruction describing a task. Write a response that appropriately completes the request.

### Instruction:
How often must an H1 model be revalidated?

### Response:
Every six months. That interval is a ceiling rather than a target...<|end_of_text|>
```

If you supervise the whole sequence, you are training the model to generate **the boilerplate and
the question** as readily as the answer. It learns that a plausible thing to emit is
`### Instruction:` followed by a question it invents. This is a real, observed failure mode.

Set `labels = -100` across the prompt span:

```
input_ids :  [Below is ... ### Response:\n]  [answer tokens]  [EOS]
labels    :  [-100 -100 ...          -100 ]  [answer tokens]  [EOS]
             └───── read, not scored ─────┘  └── supervised ──┘
```

The model still *reads* the prompt — it's in `input_ids` and attention sees it. It just isn't
*scored* on reproducing it.

Notebook 02 prints the decoded masked and supervised spans so you can see the boundary, and
asserts the supervised span aligns with the tail of `input_ids`.

> TRL's `DataCollatorForCompletionOnlyLM` does this for you by locating a response template
> string. Worth using — after you've written it once by hand.

---

## 6. EOS: the bug that looks like a decoding problem

Omit the end-of-sequence token from your training targets and you get a model that answers
correctly and then **keeps going** — inventing follow-up questions and answering those, until it
hits `max_new_tokens`.

Teams reliably misdiagnose this as a sampling problem and spend days on `temperature`, `top_p`,
and `repetition_penalty`. No decoding parameter can fix it. The model has never seen a response
*end*, so it has learned that responses don't.

```python
answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
answer_ids = answer_ids + [tokenizer.eos_token_id]      # ← this line
```

Then at generation time pass `eos_token_id=tokenizer.eos_token_id` so `generate` actually stops.
Notebook 02 measures generated length per stage, which makes "it learned to stop" a number rather
than an impression.

---

## 7. The pad token, and padding side

**Llama 3.2 ships a real pad token**: `<|finetune_right_pad_id|>`, id `128004`. The reflex
`tokenizer.pad_token = tokenizer.eos_token` is inherited from Llama 2, which had none — and it is
actively harmful here, because it makes padding and end-of-sequence the same token. You then
cannot teach the model to emit EOS without also teaching it to emit padding.

Both notebooks assert `pad_token_id != eos_token_id`.

**Padding side**: Llama 3.2's config sets `padding_side="left"`, correct for *batched generation*
(so all sequences end at the same position, where generation continues). For *training* you want
`"right"`. Both notebooks flip it explicitly.

---

## 8. fp16, bf16, and `Attempting to unscale FP16 gradients`

A T4 is Turing (sm_75) and has **no native bfloat16**. bf16 needs Ampere (sm_80+). So everything
here is fp16.

Note: `torch.cuda.is_bf16_supported()` has historically returned `True` on Turing, where bf16 is
emulated and slow. Both notebooks check compute capability directly:

```python
SUPPORTS_BF16 = torch.cuda.get_device_capability()[0] >= 8
```

**Then the error everyone hits.** Load the model in fp16, set `fp16=True`, and training dies with:

```
ValueError: Attempting to unscale FP16 gradients.
```

Mixed-precision training keeps an fp32 master copy of trainable parameters; the gradient scaler
refuses to unscale gradients that are themselves fp16. Loading in fp16 makes the LoRA parameters
fp16 too.

The fix keeps base weights in fp16 (where the memory is) and upcasts **only the trainable**
parameters:

```python
for _, p in model.named_parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()
```

This is what `prepare_model_for_kbit_training` does for QLoRA. Both notebooks do it explicitly so
it's visible rather than magic.

---

## 9. Evaluation

### Capture the baseline *before* you train

A reported perplexity of 8.4 after training means nothing. The question is what it was before.
Notebook 01 measures baseline perplexity and baseline generations before a single gradient step.

This is not a hypothetical omission — it is the most common flaw in fine-tuning write-ups.

### Perplexity, and what it doesn't tell you

$$\text{PPL} = \exp\left(\frac{1}{N}\sum_i -\log p(x_i \mid x_{<i})\right)$$

Roughly, "how many tokens is the model effectively choosing between." Lower is less surprised.

Two things to be careful about:

1. **Batch re-weighting.** HF returns the *mean* loss per batch. Averaging batch means directly is
   wrong when batches have different numbers of supervised positions. Weight by the token count:
   `(labels[:, 1:] != -100).sum()` — the shift is because position $i$ predicts $i+1$.
2. **It's in-domain by construction.** Notebook 01 splits at block level, so held-out blocks come
   from the same documents. That measures *in-domain fit*, not generalisation to unseen documents.
   For the latter you'd hold out whole documents. Both are legitimate; conflating them isn't.

### Check what you might have broken

A fine-tune that improves its target while degrading everything else has **traded**, not gained.
Notebook 01 measures perplexity on out-of-domain text with the adapter on and off. Some regression
is the normal cost of specialising; a large jump means your learning rate or epoch count is too
aggressive.

### Memorization probes

Because the corpus describes a **fictional** institution, its vocabulary is provably absent from
pretraining. If the adapted model completes "an H1 model must be revalidated every" with "six
months", that string cannot have come from anywhere but our training data.

This is a far crisper signal than a perplexity delta of a few tenths, and it's why the corpus was
built around invented terminology in the first place.

---

## 10. When *not* to fine-tune

Sensible order for closing a capability gap:

1. **Better prompting** — an afternoon.
2. **Better retrieval** — a week.
3. **A more capable base model** — a config change.
4. **Fine-tuning** — a training pipeline, a data lineage obligation, and a permanent maintenance
   burden.

Fine-tuning earns its place for **format and behaviour** (reliable structure, specific register),
**domain vocabulary** (the model mishandles your terminology), and **efficiency** (a small
fine-tuned model matching a large prompted one).

**Fine-tuning is the wrong tool for facts that change.** Weights don't update when your policy
does; the model will state last quarter's version confidently and without a citation. That belongs
in retrieval. This is the most common bad reason teams reach for fine-tuning.

Also: benchmark against **prompting alone**, not just against the raw base model. Beating the raw
base proves training did *something*, which was never in doubt. The real question is whether it
did something prompting couldn't have done more cheaply and reversibly.

---

## 11. Bugs in the reference notebooks

This repo is built from [sunnysavita10/Complete-LLM-Finetuning](https://github.com/sunnysavita10/Complete-LLM-Finetuning)
(folders 14 and 15). The overall arc is right and worth following. These specific defects are not.

### 11.1 The adapter is never loaded

```python
model_path = "/content/tinyllama-lora/checkpoint-5"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")   # ← wrong
```

`AutoModelForCausalLM` doesn't know what a LoRA adapter is. It reads the base model reference and
returns **the base model**, silently ignoring `adapter_model.safetensors` sitting right there. So
every "fine-tuned" inference cell in the reference is running the untouched base model.

Correct:

```python
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.float16)
model = PeftModel.from_pretrained(base, adapter_path)
model = model.merge_and_unload()      # optional; needed before stacking a second adapter
```

The reference *does* contain this pattern — in a **commented-out cell** in notebook 15. The
working code path doesn't use it.

**How to catch this yourself:** snapshot a weight tensor before and after and assert it changed.
Notebook 02 does exactly that. One assertion, catches it every time.

### 11.2 Training on padding

`padding="max_length"` with `labels = input_ids.copy()`. See §4.

### 11.3 Supervising the prompt

The instruction stage computes loss over the whole sequence. See §5.

### 11.4 No EOS, no eval split, no baseline

Targets have no EOS (§6), there is no held-out set, and no metric is captured before training —
so there is no evidence any of it worked.

### 11.5 Smaller things

- `load_in_8bit=True` then LoRA, without `prepare_model_for_kbit_training`.
- The tokenizer is loaded from `TinyLlama-1.1B-Chat-v1.0` while the model is
  `TinyLlama-1.1B-intermediate-step-1431k-3T` — different checkpoints of the same family.
- Checkpoint paths hardcoded to `checkpoint-5` / `checkpoint-3`, which only exist for one
  particular step count.

None of this makes the reference worthless — it's a useful tour of the landscape. But copying it
verbatim gives you a pipeline that appears to work and mostly doesn't, which is worse than one
that fails loudly.

---

## 12. Synthetic instruction data

This repo generates instruction data by **hand-authoring seeds and expanding programmatically**
(`scripts/generate_instructions.py`). The seeds are grounded in the corpus; the expansion applies
request-form decorators, plus classification tasks derived from document structure.

That is the self-instruct pattern with the LLM step replaced by templating: reproducible, free,
offline — and the paraphrases are mechanical rather than natural.

**What an LLM-driven expansion would add**, and what it needs care with:

| Gain | Care |
|---|---|
| Natural, varied phrasings | Costs money; output varies between runs |
| Much larger sets from the same seeds | Must dedup against seeds and against itself — near-duplicates inflate apparent size |
| Harder, more diverse task types | Check the licence: some model terms prohibit training on their outputs |
| Answers you didn't have to write | Generated answers need verification against source, or you're training on the generator's errors |

### Include abstention examples

If every training example has a confident answer, the model learns that every question has one.
It then answers confidently when it should decline — the worst failure mode for anything
retrieval-adjacent, and one installed by the *absence* of data rather than by anything you did.

~10% abstention is a reasonable target. This repo sits at ~4%, which is a compromise given the
small total and is the first thing to fix if you extend it.

---

## 13. Things worth trying next

| Change | Why |
|---|---|
| **More data, both stages** | Highest return by a wide margin. Everything here is data-starved. |
| **Document-level split in stage 1** | Measures generalisation to unseen documents rather than in-domain fit. |
| **`r` and `target_modules` sweep** | Both notebooks are set up so this is a one-line change. |
| **TRL `SFTTrainer`** | Handles packing and completion-only masking. Use it once you know what it's doing. |
| **QLoRA** | Necessary at 7B+. Note the merge constraint in §3. |
| **Unsloth** | ~2× faster, lower memory, for exactly this workload. |
| **DPO / ORPO** | The stage after instruction tuning. |
| **GGUF export** | `llama.cpp` conversion to run the merged model locally. |
