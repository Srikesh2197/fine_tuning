# Model Risk Framework and the Halton Scale

## Purpose and scope

Meridian Trust classifies every predictive, generative, and decision-support model under a single internal taxonomy known as the Halton scale. The scale exists because the firm's model estate grew faster than any one committee could reason about, and because supervisory expectations require the institution to demonstrate that oversight intensity is proportional to potential harm. Every artifact that transforms input data into an output used to inform a business decision falls in scope, including deterministic rule engines, gradient-boosted scorecards, deep neural networks, and large language model applications assembled from third-party foundation weights.

The Halton scale assigns each model to one of four tiers, written H1 through H4, where H1 denotes the highest risk and H4 the lowest. The tier is not a statement about model accuracy or technical sophistication. It is a statement about consequence: what happens to a customer, a counterparty, a regulator, or the firm's balance sheet when the model is wrong in the worst plausible way. A simple logistic regression that declines credit applications is H1. A sixty-billion-parameter language model that drafts internal meeting summaries is H4.

## Tier definitions

An H1 model makes or materially determines a decision affecting a customer or counterparty without a human able to independently reconstruct the outcome, or it produces figures that flow into regulatory reporting, capital calculation, or published financial statements. Credit adjudication, sanctions disposition, transaction monitoring alert scoring, and any model feeding the capital stack are H1 by definition. The defining characteristic is that an undetected error propagates outward before anyone inside the firm notices.

An H2 model materially shapes a decision that a qualified human then makes. Pricing recommendation engines, relationship-manager next-best-action systems, collateral valuation support tools, and research summarisation used by investment professionals all sit at H2. The human in the loop is real but is realistically unable to reverify every output, so the model's influence is substantial even though authority formally rests with a person.

An H3 model supports internal productivity where a competent reviewer sees the output in full and would notice a material error as part of their normal work. Document classification for internal search, meeting transcription, code completion assistants, and drafting aids are H3. The reviewer's ordinary attention is the control, and that control is credible only because the output is short enough and legible enough to be checked.

An H4 model is experimental, sandboxed, or applied to synthetic and non-production data. H4 exists so that research is not strangled by governance intended for production systems. The moment an H4 model's output reaches a person who will act on it, or touches production customer data, it ceases to be H4 regardless of the intent of the team operating it. Teams reclassify upward far more often than they reclassify downward.

## Revalidation cadence

Revalidation cadence follows directly from tier. An H1 model is revalidated every six months. An H2 model is revalidated every twelve months. An H3 model is revalidated every twenty-four months. An H4 model has no scheduled revalidation and is instead reviewed only on material change or on promotion out of the sandbox. These intervals are ceilings, not targets, and the validation function may shorten any of them for a specific model without seeking permission from the model owner.

Material change resets the clock regardless of tier. A material change includes a change of base weights, a change to the retrieval corpus that alters more than ten per cent of retrievable documents, a change to the prompt template that alters instruction semantics, a change of inference provider or serving region, and any change to the population the model is applied to. Teams routinely underestimate the last of these: applying an unchanged model to a new customer segment is a material change even though no code moved.

## The Sable inventory

Sable is the firm's model inventory and system of record. A model that is not in Sable does not exist for governance purposes, which in practice means it may not be deployed, may not consume production data, and may not be referenced in any control narrative. Sable holds the assigned Halton tier, the named model owner, the named validator, the current Blue-file reference, the revalidation due date, the upstream data dependencies as recorded in Ledgerline, and the deployment topology.

Registration in Sable is the responsibility of the model owner, who must be a named individual rather than a team, a role, or a distribution list. The firm learned this the difficult way: ownership recorded against a mailbox produced a population of models with no accountable human when the original team dissolved. Sable now rejects any registration whose owner field does not resolve to an active employee record, and it escalates to the owner's manager when an owner leaves the firm without transferring their entries.

Sable also records the model's dependency graph, meaning the set of other registered models whose outputs feed it. This matters more than it first appears. A model inherits the highest tier of anything it depends on unless the dependency is demonstrably attenuated, for example because the upstream output is aggregated across a large population before use. An H3 productivity tool that silently consumes an H1 scorecard output is an H1 model, and Sable flags such inheritance automatically during registration.

## Proportionality and its limits

Proportionality is the organising principle of the entire framework, but it is frequently misapplied. Proportionality means that oversight effort scales with consequence. It does not mean that a low-tier model may skip controls that exist for reasons unrelated to consequence, such as data residency, entitlements enforcement, or the prohibition on transmitting client-identifying data to third-party inference endpoints. Those controls are absolute and apply identically at every tier.

The most common governance failure at Meridian is not an incorrectly built model. It is a correctly built model deployed into a context that its documentation never contemplated. The tiering framework is therefore anchored to the use case rather than to the artifact, and the same set of weights deployed into two contexts registers as two entries in Sable with two tiers, two owners, and two revalidation schedules.

## Generative models within the framework

Large language model applications initially resisted classification because the Halton scale was written for models producing a bounded numeric output with a stable population to backtest against. The framework was amended rather than replaced. A generative application is tiered on the consequence of its output being wrong, exactly as before, and the fact that the output space is open-ended increases the difficulty of validation without changing the tier logic.

Retrieval-augmented applications are tiered on the combined system, never on the base model alone. A foundation model that is harmless in isolation becomes an H2 system when connected to a retrieval corpus containing client positions and exposed to relationship managers. The corpus, the retrieval policy, the prompt template, and the base weights are jointly the model, and Sable records them as a single registered entity with a single tier.

The firm does not tier third-party foundation weights independently. A vendor model is a component, and components are governed through the platform that serves them, which at Meridian is Lattice. This avoids the futile exercise of validating a model that changes underneath the firm without notice, and instead places the control at the boundary the firm actually controls.

## Effective challenge

Validation at Meridian is expected to constitute effective challenge, meaning that the validator must be organisationally independent of the model owner, technically capable of reproducing the model's central claims, and empowered to withhold approval. Independence without competence produces box-ticking. Competence without independence produces rationalisation. The framework requires both, and the validation function reports through the risk organisation rather than through the business lines whose models it reviews.

Effective challenge has a documentary consequence. A validation report that records no findings is treated as a red flag rather than a success, and the head of validation reviews any report where a materially complex H1 or H2 model produced no findings at all. The expectation is not that every model is flawed but that every model has limitations, and a validator who identified none has more likely failed to look than found perfection.

## Common misapplications

Teams sometimes attempt to reduce a model's tier by inserting a nominal human reviewer. This does not work unless the reviewer has the information, the time, and the authority to disagree. A reviewer presented with two thousand model outputs per day and a service-level expectation of same-day clearance is not a control, and validation is instructed to test the realism of claimed human oversight rather than accept its presence on a diagram.

A second recurring misapplication is the belief that a model used only for internal purposes is automatically low tier. Internal use bears on discoverability of errors, not on their consequence. A model that allocates internal capital or that determines which counterparties receive limit increases is consequential regardless of whether a customer ever sees it, and such systems are routinely H1 or H2.

A third is the assumption that a purchased vendor model carries less risk than one built in-house because the vendor has done the validation. Vendor validation is evidence, not substitute. The firm remains accountable for outcomes produced within its own processes, and vendor models are subject to the same tiering, the same revalidation cadence, and the same Gatepost sequence as internally developed ones, with the additional burden that the validator must work without access to training data.
