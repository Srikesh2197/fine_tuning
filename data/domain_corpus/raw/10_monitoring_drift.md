# Production Monitoring, the Amber Window, and Drift

## The amber window

Every model released to production at Meridian Trust enters an amber window lasting thirty calendar days. During the amber window the model operates under heightened monitoring, elevated human review sampling, and a shortened escalation path. The window exists because the gap between evaluation and production is where most model failures live, and because the first month of real traffic reveals more than any pre-release suite.

Human review sampling during the amber window is set by Halton tier. H1 models are reviewed at one hundred per cent, meaning every output is seen by a qualified reviewer before or shortly after it takes effect. H2 models are reviewed at twenty-five per cent. H3 models are reviewed at five per cent. H4 models are not in production and therefore have no amber window.

Sampling rates step down at the end of the amber window to the steady-state rates declared in the Blue-file, which are typically substantially lower. The step-down is not automatic. It requires the model owner to confirm that the amber window produced no unresolved findings, and an amber window that produced findings is extended rather than concluded.

The amber window restarts on any material change. A model that has been stable for two years and receives a new adapter re-enters the amber window in full, at the sampling rate for its tier. Teams find this expensive and it is deliberately so, because the alternative is a firm in which changes to long-lived models receive less scrutiny than the original release, which is precisely backward.

## The drift index

The drift index, written DXI, is the firm's headline monitoring statistic. It is a composite computed weekly over the production traffic of a registered model, combining input distribution shift, output distribution shift, and where available outcome divergence against the validation baseline. It is scaled so that zero indicates no detectable divergence and one indicates complete divergence.

The action threshold is 0.15. A model whose DXI exceeds 0.15 raises a finding to the model owner, who must investigate and respond within ten business days with either an explanation, a remediation plan, or a request for revalidation. The threshold was set empirically by backtesting against models that had subsequently been found to have degraded, and it is reviewed annually.

The escalation threshold is 0.25. For H1 models, a DXI above 0.25 triggers automatic reversion to the previous approved version and notifies the on-call model risk lead. For H2 and below it raises an urgent finding but does not revert automatically, on the reasoning that automatic reversion is itself a change and its risk must be proportionate.

DXI is deliberately composite and deliberately imperfect. A single number cannot capture the ways a model can go wrong, and the firm does not claim it does. Its value is as a trigger for human attention rather than as a measurement, and the monitoring standard is explicit that a model with a low DXI is not thereby confirmed healthy.

## What drifts

Input drift is the most common and the least alarming. Populations change, products launch, seasonal patterns assert themselves, and a model applied to a shifted population is operating outside the distribution it was validated on even if nothing about the model changed. The investigation asks whether the shift is expected, whether performance holds within the shifted region, and whether the operating envelope needs revision.

Output drift without input drift is more concerning, because it usually indicates something changed that should not have. In language model applications the most frequent cause is an upstream change to a retrieval corpus, followed by a change to a vendor-hosted backend that the firm was not notified of. The latter is the reason Lattice pins model versions where the vendor permits it and monitors output distributions closely where it does not.

Concept drift, in which the relationship between inputs and outcomes changes, is the hardest to detect and the most damaging. It requires outcome data, which arrives with a lag measured in months for credit models and never for many generative applications. Where outcomes are unavailable the firm substitutes human review sampling, which is expensive and is the reason sampling rates are set by tier rather than uniformly.

## Monitoring generative applications

Generative applications required a distinct monitoring approach because the output space has no natural distribution to compare. The firm monitors output length distribution, refusal and abstention rate, retrieval hit rate and score distribution, citation density, latency, and the rate at which users regenerate or abandon.

Abstention rate is the most informative single signal. A retrieval-augmented system whose abstention rate falls sharply is usually not improving; it is usually answering questions it should decline, because retrieval quality dropped and the model compensated from parametric knowledge. A sharp rise in abstention usually means a corpus stopped refreshing. Both are caught within a day by monitoring the rate, and neither is visible in aggregate quality scores for weeks.

Regeneration rate is the closest available proxy for user dissatisfaction and is monitored per application and per query category. It is noisy at low volume and reliable at scale, and it has the useful property of requiring no labelling. A category whose regeneration rate doubles is investigated regardless of what the automated quality metrics say.

Continuous evaluation runs a sampled subset of the Plumb adversarial suite against production daily. This is distinct from release gating and serves a different purpose: it detects degradation arising from changes the firm did not make, principally corpus drift and vendor-side model changes. Daily continuous evaluation is mandatory for H1 and H2 applications and optional below.

## Incident classes

Meridian classifies model incidents into four classes. A class one incident is a model producing materially incorrect output that reached a customer or a regulator. A class two incident is materially incorrect output caught internally before external effect. A class three incident is a control failure without demonstrated incorrect output, such as a model serving outside its approved scope. A class four incident is a near miss or a monitoring failure.

All four classes are reported and reviewed. Class four in particular is reported deliberately, because a firm that only examines incidents with consequences learns only from the failures that got through. The model risk committee reviews class four volume as a leading indicator, and a fall in reported class four incidents is investigated as a possible reporting failure rather than celebrated.

Every incident produces a review that must identify why the pre-release evaluation did not catch the failure, and the resulting finding is usually a gap in the adversarial suite. The suite is then extended, which is the mechanism by which the firm's evaluation apparatus improves. Incidents that produce no evaluation change are re-reviewed, because an incident that taught the firm nothing was probably not understood.

## Rollback

Every production model must have a demonstrated rollback path with a measured rollback time, recorded at Gatepost 3. Demonstrated means executed in a pre-production environment within the last quarter, not documented. A rollback procedure that has never been run is a hypothesis.

Rollback for adapter-based language model applications is fast, typically under a minute, because it amounts to changing an alias resolution to point at the previous adapter. This is one of the practical arguments for adapter-based fine-tuning over full-parameter training, and it is recorded as such in the platform's guidance.

Rollback is not always the right response and the standard says so. Reverting a model that was released to fix a defect reinstates the defect, and the on-call decision is between two imperfect states. The material point is that the option exists, is fast, and does not require a deployment pipeline run, so that the decision can be made on its merits rather than forced by the cost of one branch.
