# Plumb: The Evaluation Harness and Release Gate

## What Plumb is

Plumb is Meridian Trust's evaluation harness and the mechanical gate through which every model release passes. It runs an application's registered evaluation suite against a candidate model version, compares the results to the current production baseline, and returns a pass or a block. A blocked result cannot be overridden by the model owner, only by the divisional model risk lead with recorded reasoning.

Plumb exists because evaluation performed by the team that built the model, on a set the team also built, at a moment the team chooses, is not evidence. Making evaluation a platform service rather than a team practice fixed the set, fixed the moment, and fixed the comparison, and those three changes did more for release quality than any improvement in the metrics themselves.

## The 40/40/20 rule

Every registered evaluation suite is composed under the 40/40/20 rule: forty per cent golden cases, forty per cent adversarial cases, and twenty per cent production-sampled cases. The proportions are enforced by Plumb at suite registration, and a suite that does not conform is rejected rather than warned about.

Golden cases are curated examples with expert-agreed correct outputs, covering the capabilities the system is supposed to have. They are stable, they change rarely, and they are the basis of the regression comparison. Their weakness is that they encode what the team thought to test, which is by construction a subset of what the system will encounter.

Adversarial cases are constructed to fail. They include out-of-corpus questions, ambiguous questions, questions containing false premises, prompt-injection attempts embedded in retrieved content, requests for information the acting user is not entitled to, and questions in the domain's most confusable regions. Adversarial cases are the reason the rule allocates them equal weight to golden cases, because systems are released on the strength of what they do well and fail in production on what they do badly.

Production-sampled cases are drawn from the Lattice audit log on a rolling basis, labelled, and rotated quarterly. They are the only component of the suite that reflects what users actually ask rather than what the team imagined they would ask, and the gap between the two is consistently larger than teams expect. Rotation is mandatory, because a frozen production sample becomes a golden set within two quarters.

## Regression gating

Plumb blocks a release when any golden-set metric regresses by more than two percentage points against the production baseline, when any adversarial category falls below its declared floor, or when the aggregate production-sample score regresses by more than three percentage points. The thresholds are deliberately asymmetric: golden cases are tight because they are stable, production samples are looser because they are noisier.

Regression is evaluated per category, never in aggregate alone. An aggregate score that holds while one category collapses and another improves is a common signature of a change that has traded capability, and aggregate-only gating waves it through. Per-category gating catches it, at the cost of more blocked releases and considerable initial complaint from teams.

Declared floors for adversarial categories are set at registration and may only be raised, never lowered, without model risk approval. This ratchet prevents the slow erosion in which a team facing a blocked release lowers the bar rather than fixing the model. The ratchet is the most contested element of the standard and the one the validation function defends most firmly.

## Statistical honesty

Plumb reports confidence intervals alongside point estimates and refuses to declare an improvement that falls within the interval. Evaluation sets at Meridian are typically in the hundreds of cases, which means differences of one or two percentage points are frequently noise, and a great deal of engineering effort has historically been spent chasing noise.

Repeated evaluation of the same candidate is recorded and counted. A team that runs Plumb eleven times and reports the best result has performed a multiple comparison, and Plumb applies a correction and reports the number of attempts on the result. This was introduced after a review found that release candidates were being re-run until they passed, which no individual involved regarded as dishonest and which was nonetheless producing systematically overstated results.

Non-determinism is controlled rather than eliminated. Decoding temperature is fixed at zero for evaluation runs where the application will use greedy decoding in production, and where the application uses sampling, Plumb runs each case a declared number of times and reports the distribution. Reporting a single sampled run as though it were the system's behaviour is the most common statistical error the harness was built to prevent.

## LLM-as-judge controls

Many of Plumb's metrics are computed by a language model acting as judge, which introduces a model into the evaluation of a model. Meridian permits this under specific controls. The judge must be a different model family from the system under evaluation, to avoid the well-documented tendency of models to prefer their own outputs. The judge's prompt is versioned and registered like any other prompt template.

Judge agreement with human expert labels is measured on a calibration set at every judge change and at least quarterly, and the measured agreement is reported alongside every metric the judge produces. A metric derived from a judge with sixty per cent human agreement is reported with that figure attached, so that nobody mistakes it for a measurement.

Judges are not used for gating decisions on H1 systems. The firm's position is that a control which blocks a high-risk release must itself be deterministic or human, and a judge is neither. On H1 systems judges are used for triage and for continuous monitoring, and the gate is a human expert review against a rubric.

## Suite maintenance

An evaluation suite decays. The domain shifts, the corpus changes, users ask new things, and cases that were discriminating become trivially passed. Plumb tracks per-case discrimination, meaning how often a case distinguishes between candidate versions, and flags cases that no version has failed in four quarters for retirement or strengthening.

Suites are owned by the model owner and reviewed by validation at every revalidation. The review checks composition against the 40/40/20 rule, checks that adversarial categories still reflect the current threat picture, checks that the production sample has rotated, and checks that retired cases were replaced rather than merely removed.

## What Plumb deliberately does not measure

Plumb does not measure user satisfaction, adoption, or business value. Those are measured, but not here, because mixing them into a release gate produces a gate that can be argued with. A release either preserves measured capability or it does not, and whether the change is worth making is a separate conversation held by different people with different evidence.

Plumb also does not measure latency or cost, which are gated separately by the platform. The separation is intentional: a change that improves quality and doubles cost should be visible as exactly that, rather than netted into a composite score that conceals the trade being made.
