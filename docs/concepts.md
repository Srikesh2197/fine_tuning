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
preference tuning    (chosen, rejected) pairs — DPO / ORPO / RLHF              ← notebook 03
                     → a model that answers the way people prefer
```

The first three rungs all optimise the **same** objective: predict the next token. What changes is
the data and which tokens are scored. That's the whole idea, and holding onto it makes most of the
rest obvious.

**Preference tuning is where that stops being true**, which is why it is the interesting one. It has
no gold target to predict — only two candidate answers and an ordering between them — so the loss
is computed over a *margin* rather than over tokens. §10 works through what that changes.

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
association** is stored. If you want the model to absorb a vocabulary — `hepatic gluconeogenesis`,
`odds ratio`, `intention-to-treat` — the MLP is where that lands. Both notebooks target all seven
projections.

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

1. Most of each sequence is padding. On this repo's PubMed corpus, ~81% of token slots — abstract
   sections have a median of 80 tokens against a 512-token window.
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
Below is an instruction describing a task, paired with input providing further context.
Write a response that appropriately completes the request.

### Instruction:
Answer the research question using only the abstract provided. Begin your reply with
"Answer:" followed by yes, no, or maybe, then justify it in one or two sentences.

Question: Do mitochondria play a role in remodelling lace plant leaves during PCD?

### Input:
BACKGROUND: Programmed cell death (PCD) is the regulated death of cells within an organism...
RESULTS: The following paper elucidates the role of mitochondria during PCD...

### Response:
Answer: yes

Results depicted mitochondrial dynamics in vivo as PCD progresses...<|end_of_text|>
```

If you supervise the whole sequence, you are training the model to generate **the boilerplate, the
question, and the entire abstract** as readily as the answer. It learns that a plausible thing to
emit is `### Instruction:` followed by a question it invents. This is a real, observed failure mode.

On this dataset the arithmetic is brutal: the abstract is most of the sequence, so **only 12% of
tokens are response**. Without the mask, 88% of the training budget goes on regurgitating text the
model is only supposed to read.

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

### Memorization, not memorization probes

The earlier version of this repo trained on an invented corpus, which let it plant facts the base
model provably could not know. That is a neat trick and it is also a crutch: it proves the weights
moved, not that anything useful happened.

With real data you need the honest versions instead:

- **Stage 1** measures perplexity on held-out abstracts, plus an out-of-domain control, plus an
  n-gram overlap check against the true continuation of a *training* abstract. That last one is
  the memorization test proper: a model reproducing training text verbatim has stopped
  generalising. Low single-digit overlap is healthy; above ~30% means cut the epochs.
- **Stage 2** has a real metric — decision accuracy on held-out articles, against a 55%
  majority-class baseline. That is the number that matters, and unlike a perplexity delta it
  cannot be satisfied by the model simply becoming more fluent.

### Report the baseline you can actually lose to

Three baselines are worth carrying in any classification-flavoured fine-tune:

| baseline | what it rules out |
|---|---|
| **Majority class** | A model that learned the label prior and nothing else |
| **Untuned base model** | That the training did anything at all |
| **Base model + prompting** | That the training did something prompting couldn't |

The first is free and the most frequently omitted. On PubMedQA's 55/34/11 label split, a model
that answers "yes" to everything scores 55%, which looks respectable in isolation. Notebook 02
prints a confusion matrix precisely so that outcome is visible rather than flattering.

## 10. Preference tuning, and what DPO actually optimises

Stages 1 and 2 both had a gold answer to imitate. Stage 3 does not. It has a **prompt and two
candidate answers**, plus the knowledge that one is better than the other. Nothing says how much
better, and nothing says what the ideal answer would have been.

RLHF handles that by training a reward model on the comparisons, then optimising the policy against
it with PPO — two models, two training loops, and a notoriously fiddly middle. **DPO's contribution
is the observation that you never needed the reward model.** For the KL-constrained objective RLHF
optimises, the optimal policy has a closed form, and it can be rearranged so the reward is expressed
in terms of the policy itself. Substituting that back into the reward model's own loss leaves a
plain binary classification loss over pairs:

$$\mathcal{L}_\text{DPO} = -\log \sigma\!\left(\beta\left[\log\frac{\pi_\theta(y^+\!\mid x)}{\pi_\text{ref}(y^+\!\mid x)} - \log\frac{\pi_\theta(y^-\!\mid x)}{\pi_\text{ref}(y^-\!\mid x)}\right]\right)$$

Read it as: *raise the log-probability the model assigns to the better answer, relative to what the
frozen starting model assigned it, by more than you raise it for the worse answer.* One model, one
loop, no sampling during training.

### The reference model, and why it is not free by accident

$\pi_\text{ref}$ is the frozen model you started from. It appears in the loss twice, so every step
needs its log-probabilities as well as the policy's, and the obvious implementation keeps a second
full copy of the weights in memory. On a 16 GB T4 with a 1B model that is the difference between
fitting and not.

With LoRA it costs almost nothing, because the base weights are frozen and shared: the reference is
the *same* weights with the adapter contribution removed. Worth knowing exactly how TRL implements
that, because the usual description is wrong. Given a PEFT model and `ref_model=None`, trl 0.29
**adds a second adapter named `ref`** — a frozen copy of yours, taken when the trainer is
constructed — rather than toggling your adapter off at each reference forward. The distinction
matters in three places:

- the model now carries two adapters, so `save_pretrained()` will write both unless you pass
  `selected_adapters=["default"]`
- the reference is fixed at *construction* time, so a partly-trained adapter would be silently
  baked in as the anchor
- since LoRA initialises $B$ to zero, a fresh adapter makes the copy the identity, and the
  reference is therefore exactly the model you merged — which is the only reason the arrangement
  is sound

Notebook 03 asserts all of this rather than trusting it, and the cheapest assertion is the
strongest: **at step 0 the reward margin must be exactly zero.** If policy and reference are the
same function, every log-ratio in the loss is $\log 1$. A non-zero margin at step 0 means the
reference is not the model you think it is, and nothing downstream means anything.

### $\beta$

$\beta$ is the KL constraint, carried over from the RLHF objective DPO replaces. Low $\beta$ lets
the policy drift far from the reference and chase the preference hard; high $\beta$ keeps it close
and it learns less. As $\beta \to \infty$ nothing moves. `0.1` is the paper's value and a reasonable
first guess. It is also the one hyperparameter that changes *what* DPO does rather than how fast it
does it, so it is the first thing to sweep.

### Reading the reward curves honestly

DPO logs an implicit reward per side, $r(y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$.
It is not a quality score — it is only how much more, or less, likely the tuned model finds an
answer than the model it started from.

**`rewards/margins` will rise. That is not evidence of anything.** The margin is a difference, and
gradient descent can widen it from either end. Making text *less* likely is far easier than making
it more likely, so the optimiser overwhelmingly picks that end: `rewards/rejected` plunges,
`rewards/chosen` drifts negative too, and the margin grows the whole time. A model that has simply
learned to suppress its own output distribution produces exactly the curve people screenshot as
success.

So watch **`rewards/chosen`**, not the margin. A gentle decline is normal — especially with
off-policy preferred answers (§11). A collapse means the learning rate is too high or $\beta$ too
low. And regardless of what the curves say, the only thing that settles whether the model got
better is a task metric on held-out data, which is why notebook 03 goes back to notebook 02's
200-article accuracy test rather than declaring victory at the loss curve.

### The alternatives, in one line each

| method | what it changes |
|---|---|
| **DPO** | The baseline here: pairs, a frozen reference, a sigmoid loss over the margin. |
| **SimPO** (`loss_type=["sigmoid_norm"]`) | Length-normalises the log-probabilities, so the loss cannot be won by being shorter. The direct fix when the length audit finds a gap. |
| **IPO** | Replaces the sigmoid with a squared loss, which stops the model driving the margin to infinity on easy pairs. |
| **ORPO** | Folds the preference term into the SFT loss. No reference model and no separate stage at all. |
| **KTO** | Takes thumbs-up / thumbs-down on single answers instead of pairs. Far cheaper to collect. |
| **PPO / GRPO** | Actual RL against a reward model. What DPO is a shortcut for. |

---

## 11. Where preference pairs come from

This is the part that decides whether a preference stage teaches anything, and it gets about one
sentence in most write-ups.

### Hand-written pairs teach the API, not the model

The tutorial default is a small CSV of `(prompt, chosen, rejected)` triples written by the author —
often a handful. Two things go wrong, and they compound.

**They are trivially separable.** A rejected answer written to be obviously bad differs from the
chosen one in surface features: it is shorter, blunter, more absolute, phrased differently. DPO will
find the cheapest signal that separates the two, learn *that*, and report a beautiful reward margin.
The behaviour you cared about is untouched.

**They are off-policy.** The DPO derivation assumes the comparisons are drawn from the distribution
being optimised. Answers the model would never produce carry no information about how to reallocate
probability mass among the answers it *does* produce. You end up penalising text that already had
negligible probability, which costs a gradient step and buys nothing.

### On-policy mining

The alternative is to make the model generate its own negatives. Sample $k$ answers per training
prompt at a temperature high enough that the model disagrees with itself, grade them, and pair a
good one against a bad one. Both sides then come from the policy, and the loss is being asked the
question it was derived to answer: *of the things you actually say, which should you say more?*

This needs a grader, which is the real constraint — and it is why the dataset choice in §14 matters
for a third time. PubMedQA carries an expert `yes/no/maybe` per record, so the grader is the same
regex notebook 02 is scored with. Without a machine-checkable label you need human annotation or an
LLM judge, and the LLM judge brings its own biases (notably toward length and confidence).

Three things then need auditing before you train, because each is a way the dataset can quietly
lie:

| risk | check | why |
|---|---|---|
| **Length bias** | mean tokens, chosen vs rejected | If rejected answers skew long, DPO learns "be brief" and you report it as "learned to be correct". The most common self-deception in preference tuning. |
| **Provenance** | what fraction of `chosen` is on-policy | Falling back to an expert answer when every sample was wrong puts you back off-policy on the preferred side. Useful, but it is the usual reason `rewards/chosen` falls. |
| **Leakage** | pair articles ∩ held-out articles | Pairs mined from test articles turn the final metric into a third round of training on the test set. |

### Proxy preferences are not preferences

Worth saying plainly, because the vocabulary invites overclaiming. "Chosen" in notebook 03 means
"agreed with the expert's yes/no/maybe". That is a correctness label wearing a preference label's
clothes. Real preference data encodes helpfulness, tone, hedging, appropriate refusal, willingness
to say "I don't know" — none of which a regex can grade, and most of which is why RLHF exists.

What the pipeline genuinely demonstrates is the *mechanism*: how comparisons become a loss, how the
reference policy is arranged, and how to tell a real improvement from a moved number. Scaling it to
preferences worth the name means replacing the grader, not the trainer.

---

## 12. When *not* to fine-tune

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

## 13. Bugs in the reference notebooks

This repo is built from [sunnysavita10/Complete-LLM-Finetuning](https://github.com/sunnysavita10/Complete-LLM-Finetuning)
(folders 14, 15 and 16). The overall arc is right and worth following. These specific defects are
not.

### 13.1 The adapter is never loaded

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

### 13.2 Training on padding

`padding="max_length"` with `labels = input_ids.copy()`. See §4.

### 13.3 Supervising the prompt

The instruction stage computes loss over the whole sequence. See §5.

### 13.4 No EOS, no eval split, no baseline

Targets have no EOS (§6), there is no held-out set, and no metric is captured before training —
so there is no evidence any of it worked.

### 13.5 Smaller things

- `load_in_8bit=True` then LoRA, without `prepare_model_for_kbit_training`.
- The tokenizer is loaded from `TinyLlama-1.1B-Chat-v1.0` while the model is
  `TinyLlama-1.1B-intermediate-step-1431k-3T` — different checkpoints of the same family.
- Checkpoint paths hardcoded to `checkpoint-5` / `checkpoint-3`, which only exist for one
  particular step count.

### 13.6 The preference stage (folder 16)

Folder 16's `Preference_Aligned_Training_DPO_final.ipynb` repeats the pattern, with defects specific
to DPO:

- **Five hand-written preference pairs**, in `pharma_preference_data.csv`. Every objection in §11
  applies: hand-authored, trivially separable, entirely off-policy. Five is also fewer than one
  optimizer step at the notebook's own `gradient_accumulation_steps=8`.
- **`load_in_8bit=True` immediately before `merge_and_unload()`.** The base is loaded quantized,
  then the instruction adapter is merged into it. Merging a fp16 update into 8-bit weights is
  exactly the operation §3 says not to perform, and it is why this repo does not quantize at all.
- **`tokenizer.pad_token = tokenizer.eos_token`**, the aliasing §7 covers, and here it lands in a
  stage that is already prone to unlearning EOS.
- **No eval split and no metric.** All five pairs are used for training, `eval_dataset` is never
  passed, and the before/after comparison is a single prompt generated with `do_sample=True,
  temperature=0.8` — so consecutive runs of the *same* model disagree, and nothing distinguishes a
  training effect from the sampler.
- **`loss_type="sigmoid"` as a string**, against an unpinned `!pip install -U trl`. Since trl 0.29
  the field is a `list[str]`.

None of this makes the reference worthless — it's a useful tour of the landscape. But copying it
verbatim gives you a pipeline that appears to work and mostly doesn't, which is worse than one
that fails loudly.

---

## 14. Choosing a dataset for multi-stage fine-tuning

The hardest practical constraint in this repo was not the code — it was finding **one authentic
source that feeds every stage**. Stage 1 needs raw text; stage 2 needs instruction pairs; stage 3
needs a gradable label, so that preference pairs can be mined rather than hand-written. Most
public datasets give you one or the other, and stitching two unrelated ones together means stage 1
adapts to a domain stage 2 never asks about.

What to look for, in rough priority order:

**1. Is the raw text actually out of the base model's comfort zone?**
This is the criterion people skip, and it silently ruins the demo. SQuAD contexts, Dolly contexts,
and most "clean" corpora are Wikipedia — which every base model has already seen many times.
Domain-adapting on them moves perplexity by almost nothing, and you conclude your pipeline is
broken when it is working perfectly on data with no headroom. Specialised registers work:
biomedical abstracts, legal opinions, patents, clinical notes, code in a niche language.

**2. Does stage 2 have a gradable target?**
Free-text instruction data leaves you comparing generations by eye, which is unfalsifiable. A
label — a class, a number, a span — gives you accuracy against a baseline. PubMedQA's
`final_decision` is why notebook 02 has a real metric instead of a vibes table.

**3. Are the stages disjoint?**
If stage 1's raw text includes the articles stage 2 is evaluated on, stage 2's score measures
memorisation. PubMedQA's configs are disjoint by construction; the notebooks assert it anyway.

**4. Is it ungated, permissively licensed, and small enough?**
Gated datasets mean token setup for anyone you share the notebook with. Check the licence covers
what you're doing — some prohibit training. And a config you can pull in seconds beats one that
eats your Colab session before training starts.

### What was considered here

| candidate | both stages? | verdict |
|---|---|---|
| **`qiaojin/PubMedQA`** | ✅ `pqa_unlabeled` raw + `pqa_labeled` pairs | **chosen** — specialised text, gradable yes/no/maybe, disjoint configs, MIT |
| `rajpurkar/squad` | ✅ contexts + Q/A | Contexts are Wikipedia — no stage-1 headroom |
| `databricks/databricks-dolly-15k` | ~ | Only half the records have context, and it's Wikipedia again |
| `armanc/scientific_papers` | ❌ raw only | Would need a second dataset bolted on |
| `sahil2801/CodeAlpaca-20k` | ❌ pairs only | Same problem, other direction |

### On synthetic data

An earlier version of this repo used a hand-authored fictional corpus with invented terminology.
That has one genuine advantage — planted facts prove the weights moved, because the base model
provably could not know them — and one fatal flaw: **it tells you nothing about whether the
pipeline works on real data**, which is the whole point of building it.

Synthetic instruction data is still a legitimate and common technique, especially the
seed-and-expand pattern self-instruct popularised: hand-write ~100 seeds, expand with an LLM.
If you go that route, two things need care. Dedup aggressively — near-duplicates inflate apparent
dataset size without adding signal. And check the generating model's licence, since some terms
prohibit training on their outputs.

**One thing worth carrying over regardless of your data source:** include examples where the right
answer is "I don't know" or "the source doesn't say". A model trained only on confidently-answered
questions learns that every question has an answer, then answers confidently when it should
decline. That failure is installed by the *absence* of data, not by anything you did.

---

## 15. Things worth trying next

| Change | Why |
|---|---|
| **`pqa_artificial`** | 211k more auto-labelled records for stage 2. Highest return by a wide margin — 800 examples is very few. |
| **More abstracts in stage 1** | `N_ABSTRACTS` is 1,000 out of 61,249 available. Raise it until the Colab session is the limit. |
| **Article-level split in stage 1** | Measures generalisation to unseen papers rather than in-domain fit. |
| **A classification head** | For yes/no/maybe alone, `AutoModelForSequenceClassification` beats generation and is far cheaper to score. |
| **`r` and `target_modules` sweep** | Both notebooks are set up so this is a one-line change. |
| **TRL `SFTTrainer`** | Handles packing and completion-only masking. Use it once you know what it's doing. |
| **QLoRA** | Necessary at 7B+. Note the merge constraint in §3. |
| **Unsloth** | ~2× faster, lower memory, for exactly this workload. |
| **A $\beta$ sweep for stage 3** | The one DPO hyperparameter that changes *what* is optimised rather than how fast. See §10. |
| **ORPO or KTO** | ORPO needs no reference model; KTO needs no pairs. Both are cheaper than DPO. See §11. |
| **GGUF export** | `llama.cpp` conversion to run the merged model locally. |
