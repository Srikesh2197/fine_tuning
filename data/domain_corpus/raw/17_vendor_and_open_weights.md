# Vendor Models, Open Weights, and Third-Party Risk

## The two supply routes

Meridian Trust obtains language model capability through two routes: hosted vendor endpoints and self-hosted open-weight models. Both are governed, and the governance differs because the risks differ. A hosted endpoint places data outside the perimeter and places model behaviour outside the firm's control. A self-hosted open-weight model keeps both inside and places the operational burden on the firm.

The choice between them is made per capability class rather than as a firm-wide position. Classes involving client-identifying data that cannot be tokenised run self-hosted. Classes requiring frontier capability that the firm cannot reproduce run hosted, with Scrim tokenisation and residency enforcement. Classes where a small fine-tuned model suffices run self-hosted because it is cheaper, not because it is safer.

## Assessing a vendor

Vendor assessment covers the contractual position, the operational position, and the model itself, in that order of decisiveness. A vendor whose contract permits training on the firm's inputs is not assessed further, because no operational control compensates for that term and the negotiation either succeeds or the vendor is not used.

The contractual terms the firm requires are: no training on firm inputs, defined data retention with a stated maximum, notice of model version changes with a defined notice period, version pinning for a defined support window, defined data processing locations, and audit rights proportionate to the criticality of the use. The notice and pinning terms are the ones vendors resist most and the ones that make continuous evaluation meaningful.

Operational assessment covers availability history, incident communication, capacity commitments, and the vendor's own security posture. It is conducted by the firm's third-party risk function using the same framework applied to any critical supplier, with model-specific supplements rather than a parallel process.

Model assessment is behavioural. The firm cannot inspect training data, cannot reproduce training, and cannot rule out contamination between the vendor's published evaluations and their training corpus. Assessment therefore proceeds by testing extensively on the firm's own data and treating vendor-published figures as unverified claims, which they are.

## The version pinning problem

A hosted model that changes without notice is an unauthorised production change to every application using it. This is the single largest operational risk in the hosted route and the reason the firm negotiates pinning terms as a threshold requirement rather than a preference.

Where pinning is available, Lattice pins explicitly and upgrades are treated as material changes passing through Gatepost. Where it is not, the platform monitors output distributions closely and continuous evaluation runs daily rather than on the standard cadence, so that a behavioural shift is detected within a day rather than at revalidation.

Detected vendor-side changes trigger a validation review. The firm's position is that it does not matter whether the vendor considers the change minor; what matters is whether the firm's applications behave differently, and only the firm can determine that. Several detected changes have been benign and two have required application changes, which is the ratio the monitoring is calibrated for.

## Open-weight licence review

Open-weight models carry licences with varying terms, and the firm reviews every licence before adoption rather than assuming that publicly available means freely usable. The terms that matter in practice are restrictions on commercial use, restrictions on use in regulated activities, obligations to attribute, obligations to publish derivatives, and restrictions on using outputs to train other models.

The last of these is frequently overlooked and directly affects the firm's synthetic data practice. Generating instruction data from a model whose licence prohibits using outputs to train competing models is a licence breach even though no weights were copied, and the review therefore happens before generation rather than before deployment.

Licence review outcomes are recorded against the model in Sable and inherited by every adapter trained on it. An adapter trained on a base model with restrictive terms carries those terms, and the registry surfaces this so that a team reusing an adapter does not inadvertently inherit an obligation they were unaware of.

## Provenance of weights

The firm obtains open weights from the original publisher or from a small set of approved mirrors, and verifies checksums against the publisher's published values. Weights obtained from arbitrary sources are not permitted, for the straightforward reason that a modified model is indistinguishable from an unmodified one by inspection and can be made to behave badly under specific triggers.

Mirror approval exists because the original publisher's distribution is sometimes gated in ways that are impractical at the firm's scale, and because some mirrors provide quantised or format-converted variants that the firm uses. An approved mirror is one whose published artifacts have been verified against the original for a sample of releases and whose operator is a known entity.

Quantised and converted variants are treated as distinct models requiring their own evaluation, not as the same model in a different format. Quantisation changes behaviour, sometimes materially and rarely uniformly across tasks, and a model evaluated at full precision and deployed quantised has been evaluated as a different system.

## Concentration risk

The firm monitors concentration across vendors and across model families, because a capability available from a single supplier is a single point of failure regardless of that supplier's reliability. Every capability class in Lattice maintains at least one fallback of comparable capability from a different supply route, and the fallback is exercised periodically rather than assumed.

Exercising the fallback means running production traffic through it for a defined period and comparing outcomes, not merely confirming that it responds. A fallback that has never served real traffic is a configuration entry, and the firm's experience is that untested fallbacks fail in ways that are obvious in retrospect and invisible beforehand.

Concentration also applies to fine-tuned adapters. An adapter pinned to a vendor base model that the vendor withdraws is stranded, and the firm's guidance is that adapters representing significant investment should be trained against open weights the firm can retain indefinitely. This consideration has shifted several fine-tuning programmes toward self-hosted base models on grounds that had nothing to do with capability.

## Exit

Every vendor arrangement has a documented exit plan stating how the firm would migrate the affected applications, what it would migrate them to, and how long it would take. The plan is reviewed annually and is required to be credible rather than aspirational, which in practice means the alternative it names must be a model the firm has actually evaluated.

Exit planning for fine-tuned capability is materially harder than for prompted capability, because an adapter does not transfer between base models. This asymmetry is stated explicitly in the fine-tuning guidance, and it is one of the considerations weighed at Gatepost 1 when a team proposes fine-tuning a vendor-hosted model rather than an open-weight one.
