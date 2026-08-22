# Incident Reviews and What They Taught

## Purpose of this record

Meridian Trust maintains an internal record of model incidents written for engineers rather than for auditors. The auditable record exists separately and serves a different purpose. This record exists so that an engineer building a new system can read what went wrong before and recognise the shape of it in their own design.

Every entry states what happened, why the pre-release evaluation did not catch it, and what changed as a result. The middle question is the important one. An incident that produced a fix but no evaluation change will recur in a different system, because the firm learned about one instance rather than about the class.

## The stale policy answer

An H2 grounded question answering system serving the operations division began answering questions about a control procedure using the superseded version of the procedure. The superseded document had been withdrawn at source eleven days earlier. Investigation found that the corpus deletion path had been failing silently for those eleven days because a credential had expired, and the failure raised no alert because the monitoring watched refresh completion rather than propagation lag.

The pre-release evaluation did not catch it because the evaluation set was built from a corpus snapshot and every case was answerable from that snapshot. Nothing in the suite tested the behaviour of the system when the index disagreed with the source.

Three changes followed. Deletion moved to a dedicated fast path independent of the refresh cadence. Monitoring changed from refresh completion to the age of the oldest unpropagated source change. And every Plumb suite for a retrieval-augmented system acquired a mandatory freshness category, testing that the system surfaces effective dates and that superseded content is absent.

## The confident extraction

A bounded extraction model processing corporate action notices populated a maturity date field on notices that stated no maturity date. The generated dates were plausible and internally consistent with the rest of the document, and the downstream reconciliation process accepted them for six weeks before a mismatch surfaced.

The evaluation suite measured field-level accuracy on documents where the field was present, which was ninety-six per cent, and did not separately measure the model's behaviour on documents where it was absent. The absent case was fourteen per cent of production volume and had an accuracy of approximately zero.

The change was structural rather than local. Extraction schemas now require an explicit absent marker rather than permitting a null that is indistinguishable from an unattempted field, and Plumb requires null accuracy to be reported as a separate metric with its own floor for every extraction application. Several existing extraction models were found to have the same defect on review.

## The adapter that would not stop

A team fine-tuning their first instruction adapter reported that the model answered correctly and then continued, generating additional questions and answering those, sometimes for hundreds of tokens. They had raised a support request describing it as a decoding problem and had spent two days adjusting sampling parameters.

The cause was that their training targets contained no end-of-sequence marker. The model had never seen an example of a response ending, so it had learned that responses do not end. No decoding parameter can repair this, and the fix was to append the end-of-sequence token to every training target and retrain, which took an afternoon.

The platform's training templates now append the marker by default and validate its presence before a training run begins. The incident is recorded here because it is the single most common first-adapter failure, and because the symptom is confusing enough that teams reliably misdiagnose it as an inference problem.

## The diluted gradient

An adapter trained for a summarisation task produced substantially worse output than the base model despite a training loss that fell smoothly throughout. Investigation found that the training data had been tokenised with padding to a fixed maximum length and the padding had not been masked from the loss, so that roughly seventy per cent of the tokens the model was trained to predict were padding.

The loss curve looked healthy because the model was learning to predict padding very well, and the padding tokens dominated the average. The team's evaluation ran only after training completed, so there was no signal during the run that anything was wrong.

Two changes followed. The platform's tokenisation utilities mask padding by default and emit a warning when a training set's supervised token fraction falls below a threshold. And training runs now evaluate on a held-out set at every epoch rather than only at the end, so that a divergence between falling loss and flat evaluation quality is visible while the run is still in progress.

## The instruction scaffolding

An adapter trained for a question answering task began emitting the instruction template in its responses, producing outputs that started with the section header the training data used to introduce the response. The model had been trained with the loss computed over the entire sequence including the prompt, so it had learned to generate prompts as readily as answers.

The evaluation caught this immediately, which is the only reason it is recorded as a class four near miss rather than as an incident. The team recognised the symptom because it was already in this record, and the fix was to mask the prompt tokens from the loss and retrain.

The general lesson, recorded in the fine-tuning guidance, is that instruction tuning supervises the response and not the request. The model does not need to learn to produce the question; it needs to learn what to produce given the question, and supervising the question spends training capacity on a task that has no value at inference.

## The optimistic split

A retrieval evaluation reported recall at eight of ninety-one per cent, which was substantially better than any previous configuration and better than the team expected. The result did not reproduce on the validator's own test set, which returned sixty-eight per cent.

The cause was that the evaluation set had been split at chunk level rather than at document level, so that passages from the same documents appeared in both the index-tuning set and the evaluation set. The configuration had been tuned against documents it was then evaluated on.

Ledgerline now requires the splitting key to be declared explicitly at dataset registration and runs a near-duplicate check across declared splits using shingled hashing. The near-duplicate check has since flagged three further datasets, in each case a document appearing twice with trivial formatting differences rather than a deliberate error.

## The unanchored metric

A team submitted a fine-tuned model for validation reporting a perplexity of 8.4 on held-out domain text, presented as evidence that domain adaptation had succeeded. The validator asked what the base model scored on the same set. Nobody had measured it. When measured, the base model scored 8.6.

The training had run, the loss had fallen, and the resulting model was very slightly better than the one they started with, at a cost of a training pipeline, a registered corpus, and a revalidation obligation. The submission was returned, and the team subsequently found that most of the intended benefit was achievable through the chunk header change described in the Quarry standard.

Plumb now requires the baseline to be captured before training begins, from the same evaluation set, on the same infrastructure, and refuses to accept a fine-tuning submission that lacks one. The requirement costs a team ten minutes and has prevented at least four fine-tuning programmes from proceeding on the strength of an improvement that did not exist.
