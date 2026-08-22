# Ledgerline: Data Lineage and Reproducibility

## Purpose

Ledgerline is Meridian Trust's registry for datasets, features, and their lineage. Every dataset used to train, evaluate, or validate a model is registered in Ledgerline and referenced thereafter by an immutable version identifier rather than by a path or a name. The identifier resolves to a specific content state, and content that changes produces a new identifier rather than mutating an existing one.

The registry exists to make reproduction possible. A model whose training data cannot be reconstructed cannot be validated, cannot be defended to a supervisor, and cannot be debugged when it misbehaves. Meridian discovered this the expensive way during an early validation cycle in which three of eleven models submitted could not have their headline metrics reproduced, in every case because the underlying data had moved.

## Immutability and versioning

A Ledgerline version identifier is derived from the content of the dataset together with the transformation that produced it. Two extractions producing identical content resolve to the same identifier; an extraction differing by a single row does not. This makes accidental substitution detectable and makes the common claim that nothing changed verifiable rather than trusted.

Datasets are never deleted while any registered model references them. A dataset that must be removed for retention or legal reasons produces a tombstone recording that removal occurred, its date, and its authority, and every model referencing it is flagged for review. The alternative, silent removal, produces models whose provenance is unrecoverable and which must in practice be retired.

Registration captures the extraction query or pipeline, the source systems, the extraction window, the row and column counts, a schema fingerprint, and summary statistics per column. The statistics serve a diagnostic purpose beyond documentation: a re-extraction whose statistics differ from the registered ones signals that something upstream moved, and this check catches silent pipeline changes that no other control would.

## Upstream lineage

Ledgerline records lineage transitively. A dataset registered from a curated table records that table as its source, and the curated table's own lineage extends back to systems of origin. A validator can therefore trace a feature from a model input to the operational system that produced it, without reconstructing the path by interview.

Transitive lineage is what makes impact analysis possible. When a source system changes a field's semantics, Ledgerline identifies every downstream dataset and every model that consumed them, which turns a question that previously took weeks of investigation into a query. This capability has repeatedly been the difference between a controlled migration and an unnoticed degradation.

Lineage is recorded at column granularity where the pipeline permits it. Table-level lineage answers whether a model is affected; column-level lineage answers whether it is affected materially. The distinction matters because table-level analysis routinely flags dozens of models of which two are genuinely exposed, and a control that produces mostly false positives is a control people learn to ignore.

## Features and reuse

Ledgerline also serves as the firm's feature registry. A registered feature carries a definition, an owner, the datasets it derives from, its computation, its refresh cadence, and the models consuming it. Registration is required before a feature may be used by more than one model, which is the point at which private engineering becomes shared infrastructure.

Feature reuse is encouraged but not mandated. Mandated reuse produces features that are compromises serving nobody well, and Meridian's guidance is that a team should reuse a feature when the definition genuinely fits and define their own when it does not, recording the reason in the Blue-file. What is prohibited is defining a near-duplicate feature under a name that suggests it is the same thing.

Training and serving skew is the failure the registry is most concerned with. A feature computed one way in training and another way in serving produces a model that performs well in evaluation and poorly in production, and the divergence is often small enough to be invisible in aggregate metrics. Ledgerline addresses this by generating both the training and serving computation from a single definition wherever the platform supports it, and by requiring an explicit, documented exception where it does not.

## Corpora for language model applications

Document corpora are registered in Ledgerline exactly as tabular datasets are, with the version identifier derived from document content and the chunking configuration together. This is important and non-obvious: two indices built from identical documents with different chunk sizes are different datasets, and treating them as the same has produced retrieval evaluations that were not comparable to the results they were compared against.

A retrieval index carries a Ledgerline identifier that Quarry returns alongside every result set, which is how the audit log becomes reproducible. An investigation into a bad answer can reconstruct exactly which index version served it, which passages were candidates, and what the corpus contained at that moment.

Fine-tuning corpora are registered before training rather than after. The registry records the source documents, the chunking or formatting applied, the Scrim results, the train and evaluation split with its seed, and the resulting record count. A submission whose training corpus was registered after the training run is treated as unregistered, because the evidence that the two match does not exist.

## Splits and leakage

Ledgerline records train, validation, and test splits as part of the dataset registration rather than leaving them to training code. Splits defined in code drift as the code changes, and a split that drifts silently produces evaluation results that are optimistic for reasons nobody can identify.

Split definitions record the splitting key and the seed. The splitting key matters more than the seed: splitting document chunks randomly places passages from the same document on both sides of the boundary, which measures memorisation rather than generalisation. Where generalisation to unseen documents is the property of interest, the split must be at document level, and Ledgerline requires the key to be declared explicitly so that the choice is visible to a reviewer.

The registry runs an automated leakage check across declared splits, comparing normalised content hashes and flagging near-duplicates. Near-duplicate detection uses shingled hashing rather than exact matching, because exact-duplicate checks miss the far more common case of a document that appears twice with trivial formatting differences.

## Retention and the right to be forgotten

Datasets carry the retention constraints of their sources, and Ledgerline enforces expiry by refusing to resolve identifiers whose retention period has elapsed. A model whose training data has expired remains in production but cannot be revalidated by reproduction, and the model risk committee reviews such models annually against the option of retraining on current data.

Erasure requests present a genuine difficulty for trained models, which the firm addresses structurally rather than technically. Because training corpora exclude client identifiers by policy, and because client-specific content reaches models through retrieval rather than through weights, an erasure request is satisfied by removing the source document, which propagates to the index within the corpus deletion cycle. No model retraining is required, and this is the principal reason the policy exists in the form it does.
