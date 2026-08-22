# The Blue-file: Model Documentation Standard

## Origin and intent

The Blue-file is Meridian Trust's mandatory model documentation artifact. Every model registered in Sable has exactly one Blue-file, versioned alongside the model itself and stored in the same repository as the training code. The name is an accident of history, deriving from the colour of the binder used by the firm's first validation team, and it has outlasted both the binders and the team.

A Blue-file exists to let a competent stranger reconstruct the model's purpose, construction, limitations, and operating envelope without speaking to anyone who built it. That framing is deliberate and demanding. It is the standard against which submissions are judged, and it rules out the common pattern of documentation that is comprehensible only to those who already understand the system.

## Required sections

A Blue-file contains eleven sections in a fixed order: purpose and use, population and scope, data lineage, feature construction, model specification, training procedure, evaluation, limitations and known failure modes, monitoring plan, operating envelope, and change log. The order is fixed so that reviewers can navigate unfamiliar documents quickly, and so that omissions are conspicuous rather than buried.

The purpose and use section states what decision the model informs and who acts on the output. It must name the consuming process rather than the consuming team, because teams reorganise and processes persist. It also states explicitly what the model must not be used for, which is the section validators read first and model owners write last.

The population and scope section defines the set of entities the model is intended to apply to, with the operative word being intended. Documentation that describes the training population without stating the intended deployment population is incomplete, because the gap between the two is where most production failures originate.

## Data lineage

The data lineage section records every dataset consumed, identified by its Ledgerline version identifier rather than by name. Names are ambiguous and mutable, and a Blue-file that references a dataset by name is not reproducible. The section also records the extraction window, any filtering applied, the treatment of missing values, and the legal basis and retention constraints attached to each source.

Lineage must extend upstream past the immediate dataset to the systems of origin. A model trained on a curated table inherits every defect of the pipelines feeding that table, and a validator who cannot see past the curated layer cannot assess whether a suspicious pattern is signal or an artifact of an upstream join. Blue-files that stop at the curated layer are routinely returned.

## Limitations and known failure modes

The limitations section is the section Meridian's validation function regards as diagnostic of documentation quality overall. A limitations section that lists only generic caveats about extrapolation and data quality signals that the team has not investigated its own model. A strong limitations section names specific subpopulations where performance degrades, specific input patterns that produce unreliable output, and specific assumptions that would invalidate the model if violated.

For generative applications the limitations section carries additional weight because the failure modes are less enumerable. Teams are expected to document the categories of prompt where the system is unreliable, the observed rate and character of unsupported assertions, the behaviour when retrieval returns nothing relevant, and the behaviour when the user's question falls outside the corpus. Each of these must be evidenced from Plumb results rather than asserted.

The section must also record what the team looked for and did not find. A statement that the team tested for degraded performance across five named subpopulations and observed no material difference is more useful than silence, because silence is indistinguishable from not having looked.

## Operating envelope

The operating envelope states the conditions under which the documented performance holds. It covers input volume and rate, latency expectation, the freshness requirement for any retrieval corpus, the acceptable range for input distribution statistics, and the point at which the model should be considered out of envelope and its outputs treated as unreliable.

The envelope is enforceable rather than advisory. Lattice reads the declared envelope from the Blue-file and rejects requests that fall outside it, which means that an incorrectly narrow envelope produces production errors and an incorrectly wide one produces silent degradation. Teams therefore have a genuine incentive to get it right, which was the reason for making it machine-readable.

## Change log

The change log records every material change to the model with its date, its nature, the Gatepost decision that authorised it, and the resulting revalidation implication. It is append-only. Editing history is prohibited, and the repository enforces this through a pre-merge check that rejects modifications to existing change-log entries.

The change log is the artifact most often consulted during incidents, because the first question after any production anomaly is what changed. A change log that is accurate and current reduces incident diagnosis from hours to minutes, and this operational benefit is what persuaded engineering teams to maintain it after years of treating documentation as compliance overhead.

## Blue-files for language model applications

Language model applications required an extension rather than a rewrite. The model specification section for such systems records the base weights and their provenance, the serving configuration, the full prompt template with its version, the retrieval configuration including chunking strategy and index version, any fine-tuned adapters with their training data lineage, and the decoding parameters.

Prompt templates are treated as model code, not configuration. They are versioned in the repository, they appear in the Blue-file in full, and changing them is a material change that resets the revalidation clock. Teams resisted this initially on the grounds that prompts iterate quickly, which is precisely the argument for governing them, since an ungoverned prompt is an ungoverned model.

Where a system uses a fine-tuned adapter, the Blue-file must record the adapter's rank, the modules it targets, the training corpus with its Ledgerline identifier, the number of training steps, and the evaluation evidence that the adapter improved the intended behaviour without degrading unrelated capability. Adapters are small and cheap to produce, which makes them easy to proliferate without documentation, and the firm treats an undocumented adapter in production as a control breach rather than a housekeeping matter.

## Review and maintenance

A Blue-file is reviewed at every Gatepost and at every revalidation. Between those points it is the model owner's responsibility to keep it current, and currency is tested by sampling. The validation function selects a small number of production models each quarter and checks the Blue-file against the deployed reality, and material divergence is reported to the model risk committee.

The most common divergence is not falsification but decay. Nobody edits a Blue-file to make it wrong; the system moves and the document does not. Meridian addresses this by tying certain sections to machine-readable sources, so that the operating envelope, the dependency graph, and the monitoring thresholds are generated from the same configuration the platform enforces rather than transcribed by hand.

Sections that cannot be generated, principally purpose, limitations, and known failure modes, remain the model owner's prose responsibility. These are also the sections that carry the most information, which is an uncomfortable but consistent finding: the parts of documentation that can be automated are the parts that matter least.
