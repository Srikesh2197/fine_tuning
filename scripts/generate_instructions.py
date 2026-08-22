#!/usr/bin/env python3
"""Generate the synthetic instruction dataset for stage-2 fine-tuning.

Output records are Alpaca-style:

    {"instruction": ..., "input": ..., "output": ..., "kind": ..., "topic": ...}

`kind` is metadata for analysis (it is not fed to the model) and is one of:

    qa           hand-authored question/answer grounded in the corpus
    subject      classify which internal system a corpus passage describes
    section      recover the section heading a corpus passage sits under
    excerpt      hand-authored reasoning over a supplied excerpt
    abstain      question the documentation does not answer

How this is "synthetic"
-----------------------
The 100-odd QA seeds below are hand-authored and grounded in the corpus. The
*expansion* is programmatic: each seed is emitted twice, once bare and once
through a request-form decorator, which teaches the model that the same request
can arrive in different surface forms. The `subject` and `section` tasks are
derived mechanically from corpus.jsonl, where the correct answer is already
known from document structure.

This is the seed-and-expand pattern that self-instruct popularised, with the
LLM expansion step replaced by deterministic templating so the dataset is
reproducible offline and costs nothing. docs/concepts.md describes what an
LLM-driven expansion would add and where it would need care.

Usage:
    python scripts/generate_instructions.py
    python scripts/generate_instructions.py --check
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "domain_corpus" / "corpus.jsonl"
OUT_DIR = REPO_ROOT / "data" / "instruction"

SEED = 20260815
EVAL_FRACTION = 0.10

# --------------------------------------------------------------------------
# Hand-authored question/answer seeds, grounded in data/domain_corpus/raw/.
# (topic, instruction, output)
# --------------------------------------------------------------------------
QA_SEEDS: list[tuple[str, str, str]] = [
    # --- Halton scale / model risk -----------------------------------------
    ("model-risk", "What is the Halton scale?",
     "The Halton scale is Meridian Trust's internal model risk taxonomy. Every model is assigned one of four tiers, H1 through H4, where H1 is the highest risk and H4 the lowest. The tier reflects the consequence of the model being wrong in the worst plausible way, not the model's accuracy or technical sophistication."),
    ("model-risk", "What distinguishes an H1 model from an H2 model?",
     "An H1 model makes or materially determines a decision affecting a customer or counterparty without a human able to independently reconstruct the outcome, or it feeds regulatory reporting, capital calculation, or published financial statements. An H2 model materially shapes a decision that a qualified human then makes, so the human in the loop is real but cannot realistically reverify every output."),
    ("model-risk", "How often must an H1 model be revalidated?",
     "Every six months. That interval is a ceiling rather than a target, and the validation function may shorten it for a specific model without seeking the model owner's agreement."),
    ("model-risk", "Give the revalidation cadence for each Halton tier.",
     "H1 models are revalidated every six months, H2 every twelve months, and H3 every twenty-four months. H4 models have no scheduled revalidation and are reviewed only on material change or on promotion out of the sandbox. Material change resets the clock at any tier."),
    ("model-risk", "What counts as a material change for revalidation purposes?",
     "A change of base weights, a change to the retrieval corpus altering more than ten per cent of retrievable documents, a change to the prompt template that alters instruction semantics, a change of inference provider or serving region, and any change to the population the model is applied to. The last is the one teams most often miss, because no code moves."),
    ("model-risk", "Why is a model tiered on its use case rather than on the artifact?",
     "Because the most common governance failure is a correctly built model deployed into a context its documentation never contemplated. The same set of weights serving two contexts registers as two entries in Sable, with two tiers, two owners, and two revalidation schedules."),
    ("model-risk", "What is Sable and what does it record?",
     "Sable is the firm's model inventory and system of record. It holds the assigned Halton tier, the named model owner, the named validator, the current Blue-file reference, the revalidation due date, the upstream data dependencies recorded in Ledgerline, and the deployment topology. A model absent from Sable may not be deployed or consume production data."),
    ("model-risk", "Why must a Sable model owner be a named individual rather than a team?",
     "Because ownership recorded against a mailbox or a distribution list produced a population of models with no accountable human once the original team dissolved. Sable now rejects registrations whose owner does not resolve to an active employee record, and escalates to the owner's manager when an owner leaves the firm without transferring their entries."),
    ("model-risk", "How does Halton tier inheritance work across model dependencies?",
     "A model inherits the highest tier of anything it depends on, unless the dependency is demonstrably attenuated — for example because the upstream output is aggregated across a large population before use. An H3 productivity tool that silently consumes an H1 scorecard output is an H1 model, and Sable flags such inheritance during registration."),
    ("model-risk", "Can a team lower a model's Halton tier by adding a human reviewer?",
     "Only if the reviewer genuinely has the information, the time, and the authority to disagree. A reviewer facing two thousand outputs a day under a same-day clearance expectation is not a control. Validation is instructed to test the realism of claimed human oversight rather than accept its presence on an architecture diagram."),
    ("model-risk", "How are retrieval-augmented applications tiered?",
     "On the combined system, never on the base model alone. The corpus, the retrieval policy, the prompt template, and the base weights are jointly the model, and Sable records them as a single registered entity with a single tier. A harmless foundation model becomes an H2 system once connected to a corpus of client positions."),
    ("model-risk", "What does effective challenge require of a validator?",
     "Organisational independence from the model owner, the technical capability to reproduce the model's central claims, and the authority to withhold approval. Independence without competence produces box-ticking; competence without independence produces rationalisation. A validation report recording no findings on a complex H1 model is treated as a red flag rather than a success."),

    # --- Gatepost -----------------------------------------------------------
    ("gatepost", "What are the three Gatepost gates?",
     "Gatepost 1 is the design gate, held before substantial engineering effort is spent. Gatepost 2 is the pre-validation gate, at which the model is functionally complete and validation formally accepts it into its queue. Gatepost 3 is the production release gate, which authorises the model to receive production traffic."),
    ("gatepost", "Which Halton tiers must pass all three Gatepost gates?",
     "H1 and H2 models must pass all three. H3 models pass Gatepost 3 only, since design-stage review of internal productivity tooling costs more than it returns. H4 models are exempt from Gatepost entirely, which is the substantive privilege of sandbox status."),
    ("gatepost", "What goes into a Gatepost 1 submission?",
     "The business problem, the proposed modelling approach, the intended Halton tier with reasoning, the data the model will consume and the legal basis for consuming it, the population it will be applied to, and the decision the output will inform. It also states what the team will accept as evidence the model works — the most useful sentence in the document and the hardest to write."),
    ("gatepost", "Who approves at Gatepost 1, and what does approval mean?",
     "The divisional model risk lead. Approval confirms only that the proposed use is permissible, the data basis is sound, and the tier assignment is plausible. It is not an endorsement of the approach and carries no weight at later gates. About one in five submissions is returned, and most returns concern the data basis rather than the modelling."),
    ("gatepost", "What is the reproduction requirement at Gatepost 2?",
     "The validator runs the team's reproduction script on validation-controlled infrastructure, and if the headline metrics do not reproduce within a stated tolerance the submission is returned without further review. This single control has done more for the quality of the model estate than any other, because it makes undisclosed manual steps impossible to sustain."),
    ("gatepost", "Does acceptance at Gatepost 2 mean a model is approved?",
     "No. It means the submission is complete enough to review. Teams that treat Gatepost 2 acceptance as a milestone toward release consistently underestimate the validation period that follows, which for an H1 model averages seven weeks."),
    ("gatepost", "What does a Gatepost 3 submission add over Gatepost 2?",
     "The operational material: deployment topology, serving region and its data residency implications, the rollback procedure with a demonstrated rollback time, monitoring configuration including DXI thresholds, the amber-window sampling plan, and the on-call rotation that will respond to alerts."),
    ("gatepost", "Who sits on the Gatepost board?",
     "The divisional model risk lead, a representative of the validation function, the platform owner for the serving environment, and — for H1 models — a representative of compliance. The board convenes weekly. It does not re-litigate validation findings; it confirms they are closed or formally accepted and that the model can be operated and withdrawn safely."),
    ("gatepost", "Is a Gatepost 3 approval attached to the model or to the deployment?",
     "To the deployment. It authorises a specific model version serving a specific population from a specific region under a specific monitoring configuration. Changing any of those requires a fresh Gatepost 3 or a delegated approval recorded against the original, and the platform rejects configuration that does not match the approved scope."),
    ("gatepost", "What is the Gatepost delegated path for?",
     "Changes that do not alter the model's decision behaviour: infrastructure migration within an approved region, dependency patching, observability changes, and capacity adjustment. The divisional model risk lead approves these unilaterally and reports them to the board retrospectively."),
    ("gatepost", "When may the expedited Gatepost path be used?",
     "Only for defect remediation, where a production model is behaving incorrectly and the fix is understood. The on-call model risk lead may authorise release with a single approval, provided the full board reviews it within five business days. It may not be used for changes that are merely commercially urgent — the test is whether the current production state is causing harm, and opportunity cost is not harm."),
    ("gatepost", "What does Gatepost deliberately not assess?",
     "Whether the model is good. That is validation's role, and conflating the two produces a board debating hyperparameters and a validation function assuming the board covered the fundamentals. Gatepost assesses whether the model is permitted, documented, reproducible, monitored, and reversible."),

    # --- Blue-file ----------------------------------------------------------
    ("blue-file", "What is a Blue-file?",
     "The mandatory model documentation artifact at Meridian Trust. Every model registered in Sable has exactly one, versioned alongside the model and stored in the same repository as the training code. Its standard is that a competent stranger should be able to reconstruct the model's purpose, construction, limitations, and operating envelope without speaking to anyone who built it."),
    ("blue-file", "What sections does a Blue-file contain, and why is the order fixed?",
     "Eleven sections in a fixed order: purpose and use, population and scope, data lineage, feature construction, model specification, training procedure, evaluation, limitations and known failure modes, monitoring plan, operating envelope, and change log. The order is fixed so reviewers can navigate unfamiliar documents quickly and so omissions are conspicuous rather than buried."),
    ("blue-file", "What makes a strong limitations section?",
     "Naming specific subpopulations where performance degrades, specific input patterns that produce unreliable output, and specific assumptions that would invalidate the model. It should also record what the team looked for and did not find, because silence is indistinguishable from not having looked. Generic caveats about extrapolation signal that the team has not investigated its own model."),
    ("blue-file", "What is the operating envelope, and how is it enforced?",
     "It states the conditions under which the documented performance holds: input volume and rate, latency expectation, retrieval corpus freshness, acceptable input distribution ranges, and the point beyond which outputs should be treated as unreliable. Lattice reads it from the Blue-file and rejects out-of-envelope requests, so it is enforceable rather than advisory."),
    ("blue-file", "Why is the Blue-file change log append-only?",
     "Because it is the artifact consulted first during incidents, when the opening question is always what changed. Editing history is prohibited and the repository enforces this with a pre-merge check that rejects modifications to existing entries. An accurate change log reduces incident diagnosis from hours to minutes."),
    ("blue-file", "What must a Blue-file record about a fine-tuned adapter?",
     "The adapter's rank, the modules it targets, the training corpus with its Ledgerline identifier, the number of training steps, and evaluation evidence that the adapter improved the intended behaviour without degrading unrelated capability. An undocumented adapter in production is treated as a control breach, not a housekeeping matter."),
    ("blue-file", "How are prompt templates treated in the Blue-file?",
     "As model code rather than configuration. They are versioned in the repository, appear in the Blue-file in full, and changing one is a material change that resets the revalidation clock. Teams initially objected that prompts iterate quickly, which is precisely the argument for governing them: an ungoverned prompt is an ungoverned model."),
    ("blue-file", "What is documentation decay and how does the firm address it?",
     "Decay is the gap that opens when the system moves and the document does not — nobody edits a Blue-file to make it wrong. The firm ties machine-checkable sections such as the operating envelope, dependency graph, and monitoring thresholds to the same configuration the platform enforces, so they are generated rather than transcribed. Purpose, limitations, and failure modes remain prose, and carry the most information."),

    # --- Lattice ------------------------------------------------------------
    ("lattice", "What is Lattice?",
     "Meridian Trust's internal gateway for all large language model inference. Every request from every application transits it, whether bound for a self-hosted open-weight model or a third-party endpoint. No application holds a vendor API key directly, and network policy blocks egress to known inference providers from application subnets."),
    ("lattice", "Why did the firm centralise inference through a single gateway?",
     "Primarily for control rather than cost. Fifty teams calling providers independently means fifty places client data might leave the perimeter, fifty prompt-injection surfaces, and no coherent answer when a supervisor asks what the firm sends to which vendor. Lattice reduces that to one enforcement point."),
    ("lattice", "What is a model alias and why does Lattice use one?",
     "An alias such as `research-summariser-v3` resolves to a specific backend, prompt template version, decoding configuration, and redaction policy. The indirection lets the platform team migrate the underlying model without touching application code, and lets governance attach approval scope to something stable."),
    ("lattice", "What is a resolution stamp?",
     "A record returned with every Lattice response giving the alias, the concrete backend, the template version, and the adapter identifier if one was applied. It makes reconstructing what produced a given output six months later a lookup rather than an investigation, which matters during validation and during incidents."),
    ("lattice", "How does Lattice make governance structural rather than procedural?",
     "The alias registry refuses to resolve aliases whose backing model is not registered in Sable with a current Gatepost 3 approval. A team that skips Gatepost cannot deploy — not because a policy forbids it, but because the platform will not resolve their alias."),
    ("lattice", "How are Lattice quotas denominated, and why?",
     "In cost rather than in tokens, refreshed monthly and attributed to the owning cost centre. Token-denominated quotas created incentives to route to cheap models regardless of fitness, producing applications that were technically within quota and economically absurd."),
    ("lattice", "How does Lattice reduce prompt injection risk?",
     "It provides structural separation between instruction content, which comes from the registered prompt template, and data content, which comes from retrieval or the user. Applications that concatenate the two themselves before calling Lattice are in breach of platform policy, and the request schema makes the correct pattern the easy one."),
    ("lattice", "Explain the principle that model output is a recommendation and never an authority.",
     "Model output is never granted authority it did not already have: it may not trigger a payment, modify an entitlement, or initiate an outbound communication without an authenticated, authorised human action. This means a successful prompt injection produces bad advice rather than a bad transaction. It is the single most important security property of the platform."),

    # --- Quarry -------------------------------------------------------------
    ("quarry", "What is Quarry?",
     "Meridian Trust's retrieval service for language model applications. It indexes internal document corpora and returns passages relevant to a query, filtered by the entitlements of the acting user. Entitlements filtering is its defining feature and the reason the firm built it rather than adopting a general-purpose vector database."),
    ("quarry", "Why does Quarry apply entitlements as a pre-filter rather than filtering results afterwards?",
     "Because filtering after retrieval leaks information through the shape of the result set, through relevance scores, and through latency. Applying entitlements as a pre-filter on the candidate set means a passage the user may not see is never a candidate and never influences ranking."),
    ("quarry", "What is a contextual chunk header and what does it do?",
     "A synthesised header comprising the document title, the section path, and the effective date, prepended to each chunk before embedding and retained when the passage reaches the model. It costs a small amount of index size and materially improves both retrieval quality and the model's ability to attribute what it read — the highest-return change the retrieval team made."),
    ("quarry", "What are Quarry's default chunking parameters?",
     "Four hundred tokens per chunk with sixty tokens of overlap, and these are defaults rather than recommendations. Dense policy text with numbered clauses performs better smaller because the retrievable unit is genuinely small; research notes perform better larger because the argument spans paragraphs and a fragment loses the reasoning."),
    ("quarry", "Why does Quarry fuse retrievers by reciprocal rank rather than by normalising scores?",
     "Score normalisation across heterogeneous retrievers proved unstable as corpora changed, producing quality regressions that were hard to attribute. Reciprocal rank fusion is less theoretically satisfying and considerably more robust, which is the correct trade for a production system whose corpora shift weekly."),
    ("quarry", "Why does document deletion propagate faster than insertion in Quarry?",
     "Because the harm is asymmetric. A missing recent document produces an incomplete answer; a present withdrawn document produces a wrong one. A document withdrawn at source is removed on the next propagation cycle via a dedicated fast path, regardless of the corpus refresh cadence."),
    ("quarry", "What does Quarry return when nothing clears the relevance floor?",
     "An empty result, not the best of a poor set. Returning weak passages produces answers that appear grounded and are not, which is the most damaging failure mode a retrieval-augmented system has. Applications are expected to detect the empty result and decline to answer rather than fall back on parametric knowledge."),
    ("quarry", "Why are entitlements resolved per request rather than cached?",
     "Because the window between a revocation and a cache expiry is exactly the window in which a departing employee retrieves what they should not. Caching was considered for latency reasons and rejected; the cost is roughly eight milliseconds and is regarded as well spent."),
    ("quarry", "What is point-in-time retrieval for?",
     "Answering what the policy was on a given date rather than what it is now. A policy superseded last quarter is not merely stale, it is wrong, and answering from it is a compliance failure rather than a quality problem. Point-in-time retrieval is essential for anything supporting a dispute or an audit."),

    # --- Retrieval evaluation ----------------------------------------------
    ("rag-eval", "Why does the firm evaluate retrieval and generation separately?",
     "Because a retrieval-augmented system fails in two distinct ways, and an end-to-end score alone cannot tell them apart. A falling end-to-end score says something is wrong; a falling retrieval score with stable generation quality says the index refresh broke. The diagnostic value is worth maintaining separate labelled sets."),
    ("rag-eval", "What is distinct-source recall and why does it matter?",
     "It counts how many distinct source documents from the required set appear in the returned passages. Plain recall at k is satisfied by a configuration that returns eight chunks from the same document, crowding the context window while appearing to succeed. Distinct-source recall catches exactly that."),
    ("rag-eval", "How is grounding measured?",
     "By decomposing the output into atomic claims and checking each against the retrieved context, scored per claim rather than per response. Per-response scoring conceals the difference between one unsupported claim among twelve and a wholly fabricated answer, and those failures warrant very different responses."),
    ("rag-eval", "How are unsupported claims classified?",
     "As contradicted, unsupported but plausible, or unsupported and implausible. Contradicted claims — where the retrieved passage says the opposite — are treated as severe and block release at any tier. The other categories are tolerated at declining thresholds by Halton tier, with H1 systems held to the tightest standard."),
    ("rag-eval", "How does the firm evaluate abstention?",
     "With a labelled set in which a known proportion of queries are unanswerable from the corpus, reporting both the abstention rate on unanswerable queries and the false abstention rate on answerable ones. The two trade against each other, and the Blue-file must state the chosen operating point and the reasoning rather than reporting whichever rate flatters the system."),
    ("rag-eval", "What is citation drift?",
     "A response citing the correct document but the wrong passage within it, or citing the passage that motivated an earlier sentence rather than the one supporting the sentence it is attached to. It is invisible to users who do not follow the citation and corrosive to those who do, and only per-citation evaluation catches it."),
    ("rag-eval", "What is a counterfactual evaluation set and what does it detect?",
     "A set in which retrieved passages state something contradicting what the base model would otherwise say. A well-behaved system follows the passage; a system that has absorbed the domain too enthusiastically follows its parameters. It is the only reliable way to detect that a domain fine-tune has made the model prefer memory over context."),
    ("rag-eval", "What happens to abstention behaviour if instruction data contains no abstention examples?",
     "The model learns that every question has an answer, because in its training data every question did. Instruction sets for retrieval-augmented systems at Meridian are therefore required to include abstention examples, typically around one in ten records."),

    # --- Plumb --------------------------------------------------------------
    ("plumb", "What is Plumb?",
     "Meridian Trust's evaluation harness and the mechanical gate every model release passes through. It runs an application's registered evaluation suite against a candidate version, compares results to the current production baseline, and returns pass or block. A block can only be overridden by the divisional model risk lead, with recorded reasoning."),
    ("plumb", "State the 40/40/20 rule.",
     "Every registered evaluation suite must be forty per cent golden cases, forty per cent adversarial cases, and twenty per cent production-sampled cases. Plumb enforces the proportions at suite registration and rejects non-conforming suites rather than warning about them."),
    ("plumb", "What are adversarial cases in a Plumb suite?",
     "Cases constructed to fail: out-of-corpus questions, ambiguous questions, questions containing false premises, prompt-injection attempts embedded in retrieved content, requests for information the acting user is not entitled to, and questions in the domain's most confusable regions. They carry equal weight to golden cases because systems are released on what they do well and fail in production on what they do badly."),
    ("plumb", "Why must the production-sampled portion of a suite rotate?",
     "Because a frozen production sample becomes a golden set within two quarters, losing the one property that made it valuable — reflecting what users actually ask rather than what the team imagined they would ask. Rotation is quarterly and mandatory."),
    ("plumb", "What are Plumb's regression gating thresholds?",
     "A release is blocked if any golden-set metric regresses by more than two percentage points against the production baseline, if any adversarial category falls below its declared floor, or if the aggregate production-sample score regresses by more than three percentage points. Golden thresholds are tighter because those cases are stable; production samples are noisier."),
    ("plumb", "Why does Plumb gate per category rather than on an aggregate score?",
     "Because an aggregate that holds while one category collapses and another improves is the signature of a change that traded capability, and aggregate-only gating waves it through. Per-category gating catches it, at the cost of more blocked releases."),
    ("plumb", "What is the adversarial floor ratchet?",
     "Declared floors for adversarial categories are set at suite registration and may only be raised, never lowered, without model risk approval. The ratchet prevents the slow erosion in which a team facing a blocked release lowers the bar instead of fixing the model. It is the most contested element of the standard and the one validation defends most firmly."),
    ("plumb", "How does Plumb handle a team running evaluation repeatedly?",
     "It records and counts the attempts, applies a multiple-comparison correction, and reports the number of runs alongside the result. This followed a review finding that release candidates were being re-run until they passed — which nobody involved regarded as dishonest and which was systematically overstating results."),
    ("plumb", "What controls apply when a language model acts as judge?",
     "The judge must be a different model family from the system under evaluation, its prompt is versioned and registered, and its agreement with human expert labels is measured quarterly and reported alongside every metric it produces. Judges are not used for gating decisions on H1 systems, because a control blocking a high-risk release must be deterministic or human."),
    ("plumb", "What does Plumb deliberately not measure?",
     "User satisfaction, adoption, business value, latency, and cost. Those are measured elsewhere. Mixing them into a release gate produces a gate that can be argued with, and netting a quality gain against a cost increase conceals the trade being made."),

    # --- Scrim --------------------------------------------------------------
    ("scrim", "What is Scrim?",
     "The service that inspects and transforms data crossing trust boundaries at Meridian Trust. It runs inline within Lattice on every prompt and response, at index time within Quarry, and at ingestion within Ledgerline. It is a control, not a safety net: applications must not send data they are not entitled to send."),
    ("scrim", "What are Scrim's four transformation modes?",
     "Block rejects the request outright. Redact replaces the detected span with a category placeholder. Tokenise substitutes a stable surrogate reversible by an authorised service on the return path. Pass with annotation permits the value through and records that it was permitted, and is reserved for self-hosted models inside the firm's own perimeter."),
    ("scrim", "How does Scrim tokenisation work?",
     "The detected value is replaced with a format-preserving surrogate that belongs to no real entity. The mapping lives in a session-scoped vault with a short lifetime, is never written to durable storage, and is inaccessible to the inference path. Reversal happens inside Lattice on the return path, so the application sees real values and the backend never does."),
    ("scrim", "Why does Scrim tokenisation preserve format class but not value?",
     "Because models behave differently when input is malformed. A redaction that turns a name into a placeholder changes the task in ways that degrade output quality measurably, whereas a surrogate of similar structure and length leaves the task intact."),
    ("scrim", "What is the known failure mode of tokenisation?",
     "A model asked to reason about a tokenised value reasons about the surrogate. Asked whether two accounts belong to the same customer, it sees two surrogates and can only compare them as strings. Applications needing reasoning over identifier semantics must use a self-hosted backend or move the comparison outside the model."),
    ("scrim", "How is data residency enforced?",
     "Each corpus and each acting user carries a residency constraint, and Lattice will not route a request to a backend outside the permitted region. Constraints are evaluated on the union attached to the request, so a request combining data from two regions is restricted to backends permitted for both. Residency is not negotiable through the exception process."),
    ("scrim", "Why is tokenisation unavailable for training data?",
     "Because there is no return path on which to reverse it. Training corpora must therefore be redacted, or must contain only data the model is permitted to memorise — and the firm's default is that no model is permitted to memorise client identifiers."),
    ("scrim", "Which is the right home for sensitive client content: fine-tuning or retrieval?",
     "Retrieval. It keeps sensitive content behind an access check evaluated at request time. Fine-tuning bakes content into weights that are then served to everyone with access to the model, and no access check exists inside a weight matrix. This distinction is the organising principle for what may be fine-tuned on at all."),
    ("scrim", "How does the firm measure Scrim's false negatives?",
     "Through quarterly red-team exercises against a constructed corpus with known planted identifiers, rather than through production monitoring. A missed identifier that reaches a backend leaves no signal in normal telemetry, so it cannot be detected after the fact."),

    # --- Fine-tuning --------------------------------------------------------
    ("fine-tuning", "What is Meridian's preference order for closing a capability gap in a language model application?",
     "Better prompting first, then better retrieval, then a more capable base model, and only then fine-tuning. The ordering reflects cost of change: a prompt can be revised in an afternoon, a retrieval configuration in a week, and a fine-tune commits the firm to a training pipeline, a data lineage obligation, and a lasting revalidation burden."),
    ("fine-tuning", "When does fine-tuning earn its place?",
     "In three situations: format and behaviour, where the model must reliably produce a specific structure or follow a specific procedure that prompting achieves inconsistently; domain vocabulary, where the model handles firm terminology poorly enough that retrieval and generation both suffer; and efficiency, where a smaller fine-tuned model matches a larger prompted one at materially lower cost and latency."),
    ("fine-tuning", "Why is fine-tuning the wrong tool for injecting facts that change?",
     "Because policies change and weights do not change with them. A model fine-tuned on policy documents states last quarter's policy with total confidence and no citation. Facts that change belong in Quarry, where they can be updated and cited."),
    ("fine-tuning", "What is the difference between domain adaptation and instruction tuning?",
     "Domain adaptation, also called continued pre-training or non-instructional fine-tuning, trains on raw domain text with a plain next-token objective — no instructions, no prompts, no expected answers. Instruction tuning trains on paired request and response examples, with the loss typically computed only over the response. The first teaches what the domain sounds like; the second teaches what to do when addressed."),
    ("fine-tuning", "Why must domain adaptation come before instruction tuning?",
     "Because running them in the opposite order damages the instruction-following behaviour the second stage installed. Domain adaptation on raw text pulls the model back toward continuation rather than response, so the platform's training templates enforce the correct sequence."),
    ("fine-tuning", "What does domain adaptation alone produce, and why does it surprise teams?",
     "A model that writes convincingly in the firm's register and does not answer questions. Teams evaluate it by asking it questions and conclude the training failed. It did not fail; it did what it was asked. The correct evaluation is perplexity on held-out domain text and qualitative assessment of continuations, not instruction-following."),
    ("fine-tuning", "Why does the firm use low-rank adapters rather than full-parameter fine-tuning?",
     "They train on available hardware, produce artifacts small enough to version and distribute freely, can be attached and detached at serving time, and leave the base weights intact. The last point carries governance weight: a validator can compare behaviour with the adapter attached and detached on identical inputs, and cheap counterfactuals make effective challenge possible."),
    ("fine-tuning", "What LoRA rank does the firm recommend for which purpose?",
     "Rank eight suffices for format and register changes. Rank sixteen to thirty-two suits domain adaptation, where the model must absorb new vocabulary and relationships. Higher ranks rarely justify themselves on corpora of the size the firm typically works with. Teams are expected to report the ranks they tried, not just the one they kept."),
    ("fine-tuning", "Why does the choice of target modules matter more than the rank?",
     "Because adapting only the attention query and value projections suits behavioural change, while domain adaptation that must absorb factual content benefits from including the feed-forward projections, which carry a disproportionate share of factual association. Teams reporting disappointing domain adaptation results have usually adapted attention only."),
    ("fine-tuning", "Why merge the domain adapter into the base weights before training an instruction adapter?",
     "Merging produces a single clean base for the second stage and avoids the interaction effects that stacked adapters exhibit, which are difficult to predict and harder to validate. The registry records the composition, and the firm's practice is to merge rather than stack at serving time."),
    ("fine-tuning", "Why are adapters pinned to a specific base model version?",
     "Because attaching an adapter to a different base version produces unpredictable behaviour and the failure is silent. The platform prohibits it rather than discouraging it. The consequence is that base model upgrades require retraining every dependent adapter, which teams must plan for rather than discover."),
    ("fine-tuning", "What must be disclosed about synthetic training data?",
     "How it was generated, what source material it derives from, what model generated it if a model was involved, and what validation was performed on the result. Synthetic data from an external model raises the additional question of whether the firm is contractually permitted to train on that model's outputs, checked before generation rather than after."),
    ("fine-tuning", "On which three axes is a fine-tune evaluated?",
     "Did it improve the target behaviour, did it degrade anything else, and did it memorise anything it should not have. Teams reliably measure the first, frequently neglect the second, and almost never think of the third unprompted. Degradation is measured by running the pre-fine-tune Plumb suite unchanged against the new model."),
    ("fine-tuning", "How is memorisation measured after a fine-tune?",
     "By prompting with prefixes drawn from the training corpus and measuring how much of the continuation reproduces the source verbatim. A small amount is expected on a corpus seen repeatedly. Extended verbatim reproduction is a defect regardless of whether the content is sensitive, because a model that has memorised has stopped generalising."),
    ("fine-tuning", "What are the most common technical mistakes teams make when training their first adapter?",
     "Training on padded sequences without masking padding from the loss, which spends the training budget learning to predict padding. Instruction tuning without masking the prompt from the loss, which teaches the model to generate scaffolding. Omitting the end-of-sequence marker from targets, which produces a model that never stops. And evaluating without ever capturing a baseline."),

    # --- Monitoring ---------------------------------------------------------
    ("monitoring", "What is the amber window?",
     "A thirty-calendar-day period following every production release, during which the model operates under heightened monitoring, elevated human review sampling, and a shortened escalation path. It exists because the gap between evaluation and production is where most model failures live, and the first month of real traffic reveals more than any pre-release suite."),
    ("monitoring", "What are the amber-window human review sampling rates by Halton tier?",
     "H1 models are reviewed at one hundred per cent, H2 at twenty-five per cent, and H3 at five per cent. H4 models are not in production and have no amber window. Rates step down to the steady-state figures in the Blue-file only after the model owner confirms the window produced no unresolved findings."),
    ("monitoring", "What restarts the amber window?",
     "Any material change. A model stable for two years that receives a new adapter re-enters the amber window in full at its tier's sampling rate. Teams find this expensive and it is deliberately so, because the alternative gives changes to long-lived models less scrutiny than the original release."),
    ("monitoring", "What is DXI?",
     "The drift index — a composite computed weekly over a registered model's production traffic, combining input distribution shift, output distribution shift, and where available outcome divergence against the validation baseline. It is scaled so zero indicates no detectable divergence and one indicates complete divergence."),
    ("monitoring", "What are the DXI thresholds and what happens at each?",
     "The action threshold is 0.15: the model owner must investigate and respond within ten business days with an explanation, a remediation plan, or a request for revalidation. The escalation threshold is 0.25: for H1 models this triggers automatic reversion to the previous approved version and notifies the on-call model risk lead; below H1 it raises an urgent finding without automatic reversion."),
    ("monitoring", "Is a low DXI evidence that a model is healthy?",
     "No. DXI is deliberately composite and deliberately imperfect — a single number cannot capture the ways a model can go wrong. Its value is as a trigger for human attention rather than as a measurement, and the monitoring standard states explicitly that a low DXI does not confirm health."),
    ("monitoring", "Which kind of drift is hardest to detect, and why?",
     "Concept drift, where the relationship between inputs and outcomes changes. It requires outcome data, which arrives with a lag of months for credit models and never for many generative applications. Where outcomes are unavailable the firm substitutes human review sampling, which is why sampling rates are set by tier rather than uniformly."),
    ("monitoring", "Why is abstention rate the most informative monitoring signal for a generative application?",
     "Because a sharp fall usually means the system is answering questions it should decline — retrieval quality dropped and the model compensated from parametric knowledge — while a sharp rise usually means a corpus stopped refreshing. Both are caught within a day by monitoring the rate, and neither is visible in aggregate quality scores for weeks."),
    ("monitoring", "What are the four model incident classes?",
     "Class one is materially incorrect output that reached a customer or regulator. Class two is materially incorrect output caught internally before external effect. Class three is a control failure without demonstrated incorrect output, such as serving outside approved scope. Class four is a near miss or a monitoring failure."),
    ("monitoring", "Why does the firm investigate a fall in reported class four incidents?",
     "Because a firm that only examines incidents with consequences learns only from the failures that got through. Class four volume is a leading indicator, so a decline is treated as a possible reporting failure rather than celebrated as an improvement."),
    ("monitoring", "What does a demonstrated rollback path mean?",
     "Executed in a pre-production environment within the last quarter, not merely documented. A rollback procedure that has never been run is a hypothesis. For adapter-based applications rollback is typically under a minute, because it amounts to repointing an alias at the previous adapter."),

    # --- Ledgerline ---------------------------------------------------------
    ("ledgerline", "What is Ledgerline?",
     "The firm's registry for datasets, features, and their lineage. Every dataset used to train, evaluate, or validate a model is registered and referenced thereafter by an immutable version identifier rather than by a path or a name. It exists to make reproduction possible, without which a model cannot be validated or defended."),
    ("ledgerline", "How is a Ledgerline version identifier derived, and why does that matter?",
     "From the content of the dataset together with the transformation that produced it. Two extractions producing identical content resolve to the same identifier; one differing by a single row does not. This makes accidental substitution detectable and makes the claim that nothing changed verifiable rather than trusted."),
    ("ledgerline", "Why does the splitting key matter more than the random seed?",
     "Because splitting document chunks randomly places passages from the same document on both sides of the boundary, which measures memorisation rather than generalisation. Where generalisation to unseen documents is the property of interest, the split must be at document level, and Ledgerline requires the key to be declared explicitly."),
    ("ledgerline", "Why is a document corpus's identifier derived from both content and chunking configuration?",
     "Because two indices built from identical documents with different chunk sizes are different datasets. Treating them as the same has produced retrieval evaluations that were not comparable to the results they were compared against."),
    ("ledgerline", "What is training and serving skew, and how does Ledgerline address it?",
     "A feature computed one way in training and another way in serving, producing a model that evaluates well and performs poorly, with a divergence often too small to see in aggregate metrics. Ledgerline generates both computations from a single definition wherever the platform supports it, and requires a documented exception where it does not."),
    ("ledgerline", "How does the firm satisfy an erasure request for a trained model?",
     "Structurally rather than technically. Because training corpora exclude client identifiers by policy and client-specific content reaches models through retrieval, erasure is satisfied by removing the source document, which propagates to the index within the corpus deletion cycle. No retraining is required — the principal reason the policy exists in that form."),
    ("ledgerline", "Why must a fine-tuning corpus be registered before training rather than after?",
     "Because a submission whose corpus was registered afterwards is treated as unregistered — the evidence that the two match does not exist. Teams that train first and register later routinely discover that the dataset they registered is not quite the one they trained on."),
    ("ledgerline", "How does Ledgerline detect leakage between splits?",
     "With an automated check comparing normalised content hashes across declared splits using shingled hashing rather than exact matching. Exact-duplicate checks miss the far more common case of a document appearing twice with trivial formatting differences."),

    # --- Validation ---------------------------------------------------------
    ("validation", "What are the three questions validation is organised around?",
     "Is the model conceptually sound? Does the implementation match the concept? Does the model perform acceptably on data it has not seen? They are applied in order, and validators are instructed not to proceed to performance testing on a model whose conceptual basis they have not accepted."),
    ("validation", "What is a challenger model for?",
     "Not to be better, but to establish what performance is achievable so the champion's performance can be interpreted rather than merely recorded. Without a reference point a validator judging whether a metric is good enough is making an unanchored judgement, and unanchored judgements drift toward whatever the team presents."),
    ("validation", "Against which three references is a fine-tuned model benchmarked?",
     "The base model with no adaptation, which establishes what the fine-tune contributed; the base model with prompting alone, which establishes whether the fine-tune was necessary; and the previous production version where one exists, which establishes whether the change is an improvement."),
    ("validation", "Why does the validator construct the prompted baseline rather than accept the team's?",
     "Because the quality of the baseline prompt determines the outcome and the team constructing it has an interest. A weak baseline prompt is the easiest way to make a fine-tune look valuable, and it is usually not deliberate."),
    ("validation", "How are validation findings graded?",
     "As material, significant, or observational. A material finding blocks approval until remediated. A significant finding may be accepted with a remediation plan and a deadline. An observational finding is recorded and tracked but imposes no obligation. Grading is the validator's decision, appealable only to the model risk committee."),
    ("validation", "What happens to an accepted finding whose remediation plan was not executed?",
     "It becomes material at the next revalidation. This ratchet prevents the accumulation of permanently deferred obligations, and was introduced after a review found several models carrying accepted findings four years old."),
    ("validation", "How does the firm validate a third-party model it cannot inspect?",
     "By shifting emphasis from construction to behaviour: testing extensively on the firm's own data and treating vendor claims as unverified assertions. Where the vendor model is a component, validation addresses the system — establishing how it behaves when the component behaves badly, since the component will change without notice."),
    ("validation", "Who does a validation report have to be legible to?",
     "A reader three years in the future with no context and no access to anyone involved. The standard is applied literally during quality review of the validation function's own work, and reports that assume shared context are returned. A report requiring its author to interpret it has failed at its primary purpose."),

    # --- Patterns, prompts, agents, vendors ---------------------------------
    ("patterns", "Name the five approved application patterns for language model systems.",
     "Bounded extraction, grounded question answering, drafting with review, classification and routing, and supervised multi-step workflow. Each has a characteristic risk profile, evaluation approach, and failure mode, so identifying the pattern tells a reviewer most of what they need to ask."),
    ("patterns", "What is the characteristic failure of bounded extraction?",
     "Confident extraction of a field that is absent. A model asked for a maturity date on a document that states none will supply a plausible date, and the downstream system cannot tell the difference. Schemas therefore require an explicit absent marker, and null accuracy is measured separately from value accuracy."),
    ("patterns", "In which application pattern does fine-tuning most reliably pay for itself?",
     "Bounded extraction. A small model fine-tuned on a few thousand examples of the firm's own document types typically matches a much larger prompted model at a fraction of the cost and latency, with markedly better output schema stability. Several of the firm's highest-volume workloads are fine-tuned extraction models."),
    ("patterns", "Why does fine-tuning often disappoint for grounded question answering?",
     "Because teams fine-tune on their corpus expecting better answers and get a model that is more confident and less grounded. Domain adaptation helps by improving handling of firm vocabulary, which aids both query understanding and answer fluency, but it does not substitute for retrieval and should not be attempted as one."),
    ("patterns", "What is automation complacency and how is it mitigated?",
     "A reviewer editing drafts that are correct ninety-five per cent of the time stops reading carefully by the third week. Mitigations are requiring the model to surface uncertainty rather than smooth it away, requiring citation within drafts, and sampling reviewed outputs for independent quality assessment rather than trusting that review occurred."),
    ("patterns", "Why must a drafting system never produce output that could be sent without a review action?",
     "Because the control being relied upon is the reviewer's attention, and the interface should cost attention rather than save it. Applications that pre-populate a send field, default to approve, or make review a single keystroke are rejected at design review."),
    ("prompting", "What are the four zones of an approved prompt template?",
     "Role and task definition, operating constraints, provided context, and the request — in that fixed order. The zones are structurally delimited rather than separated by prose, and the platform's request schema carries them as distinct fields so Lattice can enforce the separation between instruction and data content."),
    ("prompting", "Which prompt template zone do teams most often omit, and why does it matter?",
     "The part of the operating constraints zone stating what the system must do when it cannot comply. It determines production behaviour, because the interesting cases are precisely the ones where compliance is impossible and the model must choose a failure mode."),
    ("prompting", "Which prompt instructions has the firm found do not work?",
     "Instructing the model not to hallucinate does not measurably reduce unsupported assertions. Instructing it to be concise without stating a length produces variable output. Instructing it to think carefully yields no reliable improvement where the model already reasons before answering. Vague citation instructions produce citations that look right and point nowhere."),
    ("prompting", "When should a team stop adding few-shot examples and fine-tune instead?",
     "Beyond roughly eight examples. Long example blocks consume context that retrieval could use, cost tokens on every request, and install behaviour that training installs more reliably. The threshold is guidance rather than a rule, and the reasoning matters more than the number."),
    ("agents", "How are tools classified for language model use?",
     "As read, compute, or effect. Read tools retrieve information and change nothing. Compute tools transform data without persisting results. Effect tools change state visible outside the workflow — writing to a system of record, sending a communication, or initiating a process."),
    ("agents", "May a model invoke an effect tool?",
     "No, under any circumstances. An effect tool is invoked by the application following a human decision, with the model's proposal recorded as an input to that decision. Read and compute tools may be invoked directly, subject to entitlements evaluated against the acting user."),
    ("agents", "How are loops controlled in multi-step workflows?",
     "By explicit step, wall-clock, and cost limits declared at registration and enforced by the platform, with repetition detection terminating workflows that invoke the same tool with the same arguments beyond a threshold. Limits are set from the workflow's designed depth plus a small margin, not from a generous default."),
    ("agents", "Why is context accumulation a risk in multi-step workflows?",
     "Because a tool result misinterpreted at step two is carried into later steps as established input, and no later step can question it. Workflows should be structured so each step can be validated against source material rather than against the previous step's output, with deterministic validation inserted where intermediate results have checkable structure."),
    ("vendor", "What contractual terms does the firm require from a model vendor?",
     "No training on firm inputs, defined data retention with a stated maximum, notice of model version changes with a defined notice period, version pinning for a defined support window, defined data processing locations, and audit rights proportionate to the use. Notice and pinning are the terms vendors resist most and the ones that make continuous evaluation meaningful."),
    ("vendor", "Why does the firm treat an unannounced vendor model change as an unauthorised production change?",
     "Because a hosted model that changes without notice alters the behaviour of every application using it. Whether the vendor considers the change minor is irrelevant; what matters is whether the firm's applications behave differently, and only the firm can determine that. Detected changes trigger a validation review."),
    ("vendor", "Why should adapters representing significant investment be trained against open weights?",
     "Because an adapter pinned to a vendor base model that the vendor withdraws is stranded, and an adapter does not transfer between base models. This consideration is weighed at Gatepost 1 and has shifted several fine-tuning programmes toward self-hosted base models on grounds unrelated to capability."),
    ("vendor", "Why are quantised model variants treated as distinct models?",
     "Because quantisation changes behaviour, sometimes materially and rarely uniformly across tasks. A model evaluated at full precision and deployed quantised has been evaluated as a different system, so the variant requires its own evaluation."),
]

# --------------------------------------------------------------------------
# Hand-authored excerpt tasks: the model must answer from a supplied passage.
# (instruction, input, output)
# --------------------------------------------------------------------------
EXCERPT_SEEDS: list[tuple[str, str, str]] = [
    ("According to the passage, what determines a model's Halton tier?",
     "The Halton scale assigns each model to one of four tiers, written H1 through H4, where H1 denotes the highest risk and H4 the lowest. The tier is not a statement about model accuracy or technical sophistication. It is a statement about consequence: what happens to a customer, a counterparty, a regulator, or the firm's balance sheet when the model is wrong in the worst plausible way.",
     "Consequence. The tier reflects what happens to a customer, counterparty, regulator, or the firm's balance sheet when the model is wrong in the worst plausible way — not the model's accuracy or technical sophistication."),
    ("Summarise the control this passage describes and state why it works.",
     "The reproduction requirement is enforced literally. The validator runs the script on validation-controlled infrastructure, and if the headline metrics do not reproduce within a stated tolerance the submission is returned without further review. This single control has done more for the quality of the firm's model estate than any other, because it makes undisclosed manual steps impossible to sustain.",
     "Independent reproduction at Gatepost 2: the validator re-runs the team's script on validation-controlled infrastructure, and a submission whose headline metrics do not reproduce within tolerance is returned unreviewed. It works because it makes undisclosed manual steps impossible to sustain."),
    ("Based on this passage, what should an application do when retrieval returns nothing?",
     "When retrieval returns nothing above the relevance floor, Quarry returns an empty result rather than the best of a poor set. This is a deliberate choice and one that applications must handle. Returning weak passages produces answers that appear grounded and are not, which is the most damaging failure mode a retrieval-augmented system has.",
     "Decline to answer. Quarry deliberately returns an empty result rather than weak passages, and the application must handle that by declining rather than answering from parametric knowledge — otherwise it produces answers that appear grounded and are not."),
    ("What failure does this passage describe, and what was its root cause?",
     "An adapter trained for a summarisation task produced substantially worse output than the base model despite a training loss that fell smoothly throughout. Investigation found that the training data had been tokenised with padding to a fixed maximum length and the padding had not been masked from the loss, so that roughly seventy per cent of the tokens the model was trained to predict were padding.",
     "An adapter that performed worse than its base model despite a smoothly falling training loss. The cause was padding to a fixed maximum length without masking the padding from the loss, so about seventy per cent of supervised tokens were padding — the loss fell because the model was learning to predict padding well."),
    ("From this passage, state the two thresholds and what each triggers.",
     "The action threshold is 0.15. A model whose DXI exceeds 0.15 raises a finding to the model owner, who must investigate and respond within ten business days with either an explanation, a remediation plan, or a request for revalidation. The escalation threshold is 0.25. For H1 models, a DXI above 0.25 triggers automatic reversion to the previous approved version and notifies the on-call model risk lead.",
     "DXI 0.15 is the action threshold: it raises a finding requiring the model owner to respond within ten business days with an explanation, remediation plan, or revalidation request. DXI 0.25 is the escalation threshold: for H1 models it triggers automatic reversion to the previous approved version and notifies the on-call model risk lead."),
    ("What does this passage say about the order of the two fine-tuning stages?",
     "The two are complementary and ordered. Domain adaptation first, instruction tuning second. Running them in the opposite order damages the instruction-following behaviour that the second stage installed, because domain adaptation on raw text pulls the model back toward continuation rather than response. Teams that have tried the reverse order report exactly this, and the platform's training templates enforce the correct sequence.",
     "Domain adaptation first, instruction tuning second. The reverse order damages the instruction-following behaviour the second stage installs, because training on raw text pulls the model back toward continuation rather than response. The platform's training templates enforce the correct sequence."),
    ("According to this passage, why is the limitations section diagnostic of documentation quality?",
     "The limitations section is the section Meridian's validation function regards as diagnostic of documentation quality overall. A limitations section that lists only generic caveats about extrapolation and data quality signals that the team has not investigated its own model. A strong limitations section names specific subpopulations where performance degrades, specific input patterns that produce unreliable output, and specific assumptions that would invalidate the model if violated.",
     "Because generic caveats about extrapolation and data quality signal that the team never investigated its own model, whereas naming specific degraded subpopulations, specific unreliable input patterns, and specific invalidating assumptions can only come from having looked."),
    ("What problem does this passage identify with plain recall at k?",
     "Recall at k understates a specific and common failure: the relevant passage is retrieved but ranked below several near-duplicates, crowding the context window. Meridian therefore also reports distinct-source recall, which counts how many distinct source documents among the required set appear in the returned passages. A retrieval configuration that returns eight chunks from the same document scores well on plain recall and poorly on the metric that matters.",
     "It is satisfied by configurations that return many near-duplicate chunks from a single document, which crowds the context window while scoring well. Distinct-source recall — counting distinct source documents among the returned passages — exposes that failure."),
    ("Restate the security principle in this passage in one sentence, and give its practical consequence.",
     "The residual control is that model outputs are never granted authority they did not already have. A model output may not trigger a payment, may not modify an entitlement, and may not initiate an outbound communication without a human action that is itself authenticated and authorised.",
     "Model output is a recommendation and never an authority — it cannot trigger a payment, change an entitlement, or send a communication without an authenticated, authorised human action. The consequence is that a successful prompt injection produces bad advice rather than a bad transaction."),
    ("What does this passage say a validation report must include that is usually omitted?",
     "A validation report states the scope of what was reviewed, the tests performed, the findings with their grades, the validator's opinion on residual risk, and any conditions attached to approval. It also states what was not reviewed and why, which is the section most often omitted and most useful to a supervisor.",
     "A statement of what was not reviewed and why. It is the section most often omitted from validation reports and the one a supervisor finds most useful."),
    ("Explain the asymmetry this passage describes and why it exists.",
     "Deletion propagates faster than insertion by design. A document withdrawn at source is removed from the index on the next propagation cycle regardless of the corpus refresh cadence, on a dedicated fast path. The asymmetry reflects the asymmetry of harm: a missing recent document produces an incomplete answer, while a present withdrawn document produces a wrong one.",
     "Deletions propagate on a dedicated fast path independent of the corpus refresh cadence, while insertions wait for the normal cycle. The asymmetry mirrors the harm: a missing new document yields an incomplete answer, whereas a lingering withdrawn document yields a wrong one."),
    ("What does this passage identify as the cost of getting the operating envelope wrong in either direction?",
     "The envelope is enforceable rather than advisory. Lattice reads the declared envelope from the Blue-file and rejects requests that fall outside it, which means that an incorrectly narrow envelope produces production errors and an incorrectly wide one produces silent degradation. Teams therefore have a genuine incentive to get it right, which was the reason for making it machine-readable.",
     "Too narrow and Lattice rejects legitimate requests, producing visible production errors. Too wide and out-of-envelope requests are served, producing silent degradation. Making the envelope machine-readable and enforced is what gives teams a real incentive to declare it accurately."),
]

# --------------------------------------------------------------------------
# Abstention examples: questions the internal documentation does not answer.
# --------------------------------------------------------------------------
ABSTAIN_SEEDS: list[tuple[str, str]] = [
    ("What is Meridian Trust's current share price?",
     "The internal documentation does not cover that. Meridian Trust's model risk and platform standards describe governance, tooling, and engineering practice; they contain no market or financial data."),
    ("How many employees work in Meridian Trust's model validation function?",
     "The documentation does not state headcount figures. It describes the validation function's mandate, independence, and reporting line, but not its size."),
    ("What is the Halton tier of the retail overdraft scorecard?",
     "The documentation does not record tier assignments for individual models. Those live in Sable, the model inventory. The framework itself only defines the tier criteria — a scorecard that adjudicates customer decisions would meet the H1 criteria, but the specific assignment is not something the standards document."),
    ("Which vendor supplies Meridian Trust's frontier language models?",
     "The documentation does not name vendors. It describes the assessment framework, the contractual terms the firm requires, and how hosted and self-hosted routes are governed, without identifying specific suppliers."),
    ("What learning rate should I use to fine-tune a 7B model on 200,000 tokens?",
     "The internal guidance does not specify learning rates. It covers rank selection, target module choice, the ordering of domain adaptation and instruction tuning, and evaluation obligations, but leaves optimiser settings to the training team."),
    ("What was the total cost of the Lattice platform build?",
     "The documentation does not include cost or budget figures. It describes Lattice's architecture, controls, and quota model, but not the economics of building it."),
    ("Can you give me the contents of the sanctions screening model's Blue-file?",
     "No. The documentation describes what a Blue-file must contain and how it is reviewed; it does not reproduce the Blue-files of individual models, and those would in any case be subject to their own access controls."),
    ("What is the DXI of the client onboarding classifier right now?",
     "The documentation does not carry live monitoring values. It defines DXI, its thresholds of 0.15 for action and 0.25 for escalation, and the response each requires — but current readings live in the monitoring system, not in the standards."),
    ("Who is the divisional model risk lead for the markets division?",
     "The documentation identifies roles rather than individuals. It states that the divisional model risk lead approves at Gatepost 1 and chairs delegated approvals, but does not name post-holders."),
    ("What regulatory fines has Meridian Trust received for model failures?",
     "The documentation does not cover enforcement history. The incident record it maintains is written for engineers and describes technical failures and the changes that followed, not regulatory outcomes."),
    ("How do I request an exception to the data residency constraint?",
     "The documentation states that residency constraints are not negotiable through the exception process — where a capability is required and cannot be served in-region, the answer is to bring the capability in-region. It does not describe an exception route, because there is not one."),
    ("What is the maximum context length supported by Lattice?",
     "The documentation does not state a platform-wide context limit. Limits are set per model through the operating envelope declared in each Blue-file, which Lattice reads and enforces."),
    ("Summarise Meridian Trust's approach to quantum computing.",
     "The documentation does not address quantum computing. It covers model risk governance, the language model platform, retrieval, evaluation, fine-tuning practice, monitoring, and third-party model risk."),
    ("Which open-weight model family does the firm fine-tune most often?",
     "The documentation does not name specific model families. It sets out when open weights are preferred over hosted endpoints, how licences are reviewed, and how weight provenance is verified, without identifying particular models."),
    ("What is the average time to close a material validation finding?",
     "The documentation does not report operational metrics of that kind. It defines the finding grades and states that material findings block approval until remediated, but gives no timing statistics."),
]

# --------------------------------------------------------------------------
# Request-form decorators: same meaning, different surface form.
# --------------------------------------------------------------------------


def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


DECORATORS = [
    lambda q: f"Answer concisely: {q}",
    lambda q: f"For a new joiner on the platform team — {_lower_first(q)}",
    lambda q: f"In a short paragraph, {_lower_first(q)}",
    lambda q: f"{q} Keep it brief.",
    lambda q: f"I'm reviewing a Gatepost submission. {q}",
    lambda q: f"Explain for someone who has not read the standards: {_lower_first(q)}",
]


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        sys.exit(
            f"{CORPUS_PATH} not found — run `python scripts/build_corpus.py` first"
        )
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_records(corpus: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    records: list[dict] = []

    # 1. QA seeds, each emitted bare and through one decorator.
    #    Both variants share a `_group` so the split cannot separate them —
    #    they have identical outputs, and splitting them would leak the answer
    #    from train into eval.
    for i, (topic, instruction, output) in enumerate(QA_SEEDS):
        records.append(
            {"instruction": instruction, "input": "", "output": output,
             "kind": "qa", "topic": topic, "_group": f"qa:{i}"}
        )
        decorate = DECORATORS[i % len(DECORATORS)]
        records.append(
            {"instruction": decorate(instruction), "input": "", "output": output,
             "kind": "qa", "topic": topic, "_group": f"qa:{i}"}
        )

    # 2. Hand-authored excerpt tasks.
    for i, (instruction, passage, output) in enumerate(EXCERPT_SEEDS):
        records.append(
            {"instruction": instruction, "input": passage, "output": output,
             "kind": "excerpt", "topic": "excerpt", "_group": f"excerpt:{i}"}
        )

    # 3. Subject classification derived from document structure.
    from build_corpus import DOC_SUBJECT  # noqa: PLC0415  (same-package import)

    by_doc: dict[str, list[dict]] = {}
    for row in corpus:
        by_doc.setdefault(row["doc"], []).append(row)

    for doc in sorted(by_doc):
        # Pick one reasonably long paragraph per document, deterministically.
        candidates = [r for r in by_doc[doc] if len(r["text"].split()) >= 60]
        chosen = rng.choice(candidates or by_doc[doc])
        records.append(
            {
                "instruction": "Identify which Meridian Trust internal system or standard the following passage describes. Answer with the name only.",
                "input": chosen["text"],
                "output": DOC_SUBJECT[doc],
                "kind": "subject",
                "topic": chosen["topic"],
                "_group": f"subject:{doc}",
            }
        )

    # 4. Section recovery — structural understanding of internal documents.
    section_pool = [
        r for r in corpus if r["section"] and len(r["text"].split()) >= 60
    ]
    for i, chosen in enumerate(rng.sample(section_pool, k=min(12, len(section_pool)))):
        records.append(
            {
                "instruction": "The following passage comes from a Meridian Trust internal standard. State the section heading it appears under.",
                "input": chosen["text"],
                "output": chosen["section"],
                "kind": "section",
                "topic": chosen["topic"],
                "_group": f"section:{i}",
            }
        )

    # 5. Abstention examples.
    for i, (instruction, output) in enumerate(ABSTAIN_SEEDS):
        records.append(
            {"instruction": instruction, "input": "", "output": output,
             "kind": "abstain", "topic": "abstain", "_group": f"abstain:{i}"}
        )

    return records


def normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def dedup(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    for r in records:
        key = (normalise(r["instruction"]), normalise(r["input"]))
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, len(records) - len(kept)


def split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into train/eval, stratified by `kind` and grouped by `_group`.

    The split key is `_group`, not the record, because a QA seed is emitted
    twice with the same output. Splitting those two apart would put the answer
    in train and the question in eval, which is leakage dressed up as a metric.
    Stratifying by `kind` keeps every task type represented in eval.
    """
    rng = random.Random(SEED + 1)

    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["_group"], []).append(r)

    kind_of = {g: rows[0]["kind"] for g, rows in groups.items()}
    by_kind: dict[str, list[str]] = {}
    for group, kind in kind_of.items():
        by_kind.setdefault(kind, []).append(group)

    train: list[dict] = []
    evaluation: list[dict] = []
    for kind in sorted(by_kind):
        names = sorted(by_kind[kind])
        rng.shuffle(names)
        n_eval = max(1, round(len(names) * EVAL_FRACTION))
        for name in names[:n_eval]:
            evaluation.extend(groups[name])
        for name in names[n_eval:]:
            train.extend(groups[name])

    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation


def serialise(records: list[dict]) -> str:
    return "".join(
        json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                   ensure_ascii=False) + "\n"
        for r in records
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if committed files differ from a fresh build")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    corpus = load_corpus()
    records = build_records(corpus)
    records, removed = dedup(records)
    train, evaluation = split(records)

    # Leakage guard 1: an eval prompt must not also appear in train.
    train_keys = {(normalise(r["instruction"]), normalise(r["input"])) for r in train}
    leaked = [
        r for r in evaluation
        if (normalise(r["instruction"]), normalise(r["input"])) in train_keys
    ]
    if leaked:
        sys.exit(f"{len(leaked)} eval prompts also appear in train — check dedup logic")

    # Leakage guard 2: for free-text kinds, an eval answer must not appear in
    # train either. `subject` and `section` are label-prediction tasks whose
    # answers are class labels and are expected to recur, so they are exempt.
    FREE_TEXT = {"qa", "excerpt", "abstain"}
    train_outputs = {normalise(r["output"]) for r in train if r["kind"] in FREE_TEXT}
    leaked_answers = [
        r for r in evaluation
        if r["kind"] in FREE_TEXT and normalise(r["output"]) in train_outputs
    ]
    if leaked_answers:
        sys.exit(
            f"{len(leaked_answers)} eval answers also appear in train — "
            "the split is not group-aware"
        )

    train_path = OUT_DIR / "train.jsonl"
    eval_path = OUT_DIR / "eval.jsonl"
    payloads = {train_path: serialise(train), eval_path: serialise(evaluation)}

    if args.check:
        for path, payload in payloads.items():
            if not path.exists():
                sys.exit(f"{path} does not exist")
            if path.read_text(encoding="utf-8") != payload:
                sys.exit(
                    f"{path} is stale — re-run `python scripts/generate_instructions.py`"
                )
        print("OK: instruction data matches a fresh build")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in payloads.items():
        path.write_text(payload, encoding="utf-8")

    kinds: dict[str, int] = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1

    print(f"Wrote {train_path.relative_to(REPO_ROOT)} and {eval_path.relative_to(REPO_ROOT)}")
    print(f"  total records : {len(records)} ({removed} duplicates removed)")
    print(f"  train / eval  : {len(train)} / {len(evaluation)}")
    print(f"  with input    : {sum(1 for r in records if r['input'])}")
    print("  by kind       : " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))


if __name__ == "__main__":
    main()
