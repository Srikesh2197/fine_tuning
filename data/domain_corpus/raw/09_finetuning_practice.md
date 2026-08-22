# Fine-Tuning Practice at Meridian Trust

## When fine-tuning is the right answer

Meridian Trust's default answer to a capability gap in a language model application is not fine-tuning. It is, in order of preference, better prompting, better retrieval, a more capable base model, and only then fine-tuning. This ordering reflects cost of change rather than technical merit: a prompt can be revised in an afternoon, a retrieval configuration in a week, and a fine-tune commits the firm to a training pipeline, a data lineage obligation, and a revalidation burden that persists.

Fine-tuning earns its place in three situations. The first is format and behaviour: the model must reliably produce a specific output structure, adopt a specific register, or follow a specific procedure, and prompting achieves this inconsistently. The second is domain vocabulary: the model handles the firm's terminology poorly enough that retrieval and generation both suffer. The third is efficiency: a smaller fine-tuned model matches a larger prompted one at materially lower cost and latency.

Fine-tuning is the wrong answer for injecting facts that change. Teams repeatedly propose fine-tuning on policy documents so that the model knows the policy, and this fails for a reason that is structural rather than technical. Policies change, weights do not change with them, and the resulting model states last quarter's policy with total confidence and no citation. Facts that change belong in Quarry.

## Domain adaptation versus instruction tuning

The firm distinguishes two stages that teams frequently conflate. Domain adaptation, also called continued pre-training or non-instructional fine-tuning, trains the model on raw domain text with a plain next-token objective. There are no instructions, no prompts, and no expected answers. The model simply reads the corpus and adjusts its distribution toward it.

Instruction tuning trains the model on paired examples of a request and a desired response. The objective is the same next-token prediction, but the data has structure and the loss is typically computed only over the response portion. The model learns what to do when addressed, rather than what the domain sounds like.

The two are complementary and ordered. Domain adaptation first, instruction tuning second. Running them in the opposite order damages the instruction-following behaviour that the second stage installed, because domain adaptation on raw text pulls the model back toward continuation rather than response. Teams that have tried the reverse order report exactly this, and the platform's training templates enforce the correct sequence.

Domain adaptation alone produces a model that writes convincingly in the firm's register and does not answer questions. This surprises teams who evaluate a domain-adapted model by asking it questions and conclude the training failed. It did not fail; it did what it was asked. The correct evaluation for a domain-adapted model is perplexity on held-out domain text and qualitative assessment of continuations, not instruction-following.

## Adapter-based training

All fine-tuning at Meridian uses low-rank adapters rather than full-parameter updates. The reasons are practical rather than ideological. Adapters train on available hardware, they produce artifacts small enough to version and distribute freely, they can be attached and detached at serving time, and they leave the base weights intact so that the firm can reason about what changed.

The last point carries governance weight disproportionate to its technical significance. A validator reviewing a full fine-tune must accept that every parameter moved and reason about the model as a new object. A validator reviewing an adapter can inspect exactly which modules were modified, at what rank, and can compare behaviour with the adapter attached and detached on identical inputs. The counterfactual is cheap, and cheap counterfactuals make effective challenge possible.

Adapter rank is chosen by evidence rather than convention. The firm's working guidance is that rank eight suffices for format and register changes, rank sixteen to thirty-two for domain adaptation where the model must absorb new vocabulary and relationships, and higher ranks rarely justify themselves on corpora of the size the firm typically works with. Teams are expected to report the ranks they tried.

Module selection matters more than rank in the firm's experience. Adapting only the attention query and value projections is the common default and is well suited to behavioural change. Domain adaptation that must absorb factual content benefits from including the feed-forward projections, because the feed-forward layers carry a disproportionate share of factual association. Teams reporting disappointing domain adaptation results have usually adapted attention only.

## The adapter registry

Adapters are registered in Sable as components of the model entries that use them, never as standalone artifacts. An adapter has a training corpus with a Ledgerline identifier, a base model it was trained against, a rank and module specification, a training configuration, and evaluation evidence. An adapter file present in a serving path without a corresponding registry entry is a control breach.

Adapters are pinned to the base model version they were trained against. Attaching an adapter to a different base version is prohibited by the platform, not merely discouraged, because the behaviour is unpredictable and the failure is silent. Base model upgrades therefore require retraining every dependent adapter, which is a real cost and one teams must plan for rather than discover.

The registry records adapter composition where multiple adapters apply. Meridian's practice is to merge a domain adaptation adapter into the base weights before training an instruction adapter on top, rather than stacking two adapters at serving time. Merging produces a single clean base for the second stage and avoids the interaction effects that stacked adapters exhibit, which are difficult to predict and harder to validate.

## Training data obligations

Every fine-tuning corpus is registered in Ledgerline with an immutable version identifier before training begins. Training against an unregistered dataset produces an adapter that cannot pass Gatepost 2, because reproduction is impossible. Teams occasionally train first and register afterward, and discover that the dataset they registered is not quite the one they trained on.

Corpora are passed through Scrim before training, and the results are recorded. The recorded result is not merely that Scrim ran, but what it found and what was done about it, because a corpus with zero detections is either clean or incorrectly configured and the distinction matters.

Synthetic training data is permitted and is subject to an additional disclosure obligation. The Blue-file must record how the synthetic data was generated, what source material it was derived from, what model generated it if a model was involved, and what validation was performed on the result. Synthetic instruction data derived from the firm's own documents is the most common pattern and is generally uncontroversial; synthetic data generated by an external model carries the additional question of whether the firm is permitted to train on that model's outputs, which is a contractual matter checked before generation rather than after.

## Evaluating a fine-tune

A fine-tune is evaluated on three axes: did it improve the target behaviour, did it degrade anything else, and did it memorise anything it should not have. Teams reliably measure the first, frequently neglect the second, and almost never think of the third without being prompted.

Degradation is measured by running the pre-fine-tune Plumb suite unchanged against the fine-tuned model. A fine-tune that improves its target metric while regressing three unrelated categories has traded capability, and whether that trade is acceptable is a decision for the model owner and the validator jointly rather than a detail to be omitted from the submission.

Memorisation is measured by prompting with prefixes drawn from the training corpus and measuring how much of the continuation reproduces the source verbatim. A small amount of reproduction is expected and unremarkable on a corpus the model has seen repeatedly. Extended verbatim reproduction indicates over-training and is treated as a defect regardless of whether the memorised content is sensitive, because a model that has memorised is a model that has stopped generalising.

## Common mistakes

The most frequent technical mistake is training on padded sequences without masking the padding from the loss, which dilutes the gradient with tokens that carry no information and produces a model that has spent a substantial share of its training budget learning to predict padding. The second is instruction tuning without masking the prompt from the loss, which teaches the model to generate the instruction scaffolding rather than to answer.

The third is omitting an end-of-sequence marker from instruction training targets, which produces a model that answers correctly and then continues indefinitely, inventing further questions and answering those too. The symptom is unmistakable once recognised and thoroughly confusing before, and it accounts for a substantial fraction of the support requests the platform team receives from teams training their first adapter.

The fourth is evaluating the fine-tuned model without ever establishing a baseline. A team that reports a perplexity of 8.4 after training has reported nothing, because the question is what it was before. Plumb requires the baseline to be captured before training begins, from the same evaluation set, on the same infrastructure.
