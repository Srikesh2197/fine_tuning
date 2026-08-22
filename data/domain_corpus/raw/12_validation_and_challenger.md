# Independent Validation and Challenger Models

## The validation mandate

Validation at Meridian Trust is performed by a function that reports through risk rather than through the business lines whose models it reviews. Validators are not consulted during development, do not advise on model design, and do not carry any accountability for the model's business performance. The separation is uncomfortable for everyone involved and is the entire point.

A validator's mandate is to determine whether the model does what the Blue-file claims, whether the claims are the right ones, whether the limitations are complete, and whether the residual risk is acceptable at the model's Halton tier. A validator may withhold approval, may impose conditions, and may require a tier reassignment. A validator may not be overruled by the model owner, only by the model risk committee with recorded reasoning.

## The three questions

Validation is organised around three questions, applied in order. Is the model conceptually sound? Does the implementation match the concept? Does the model perform acceptably on data it has not seen? Failing the first makes the other two irrelevant, and validators are instructed not to proceed to performance testing on a model whose conceptual basis they have not accepted.

Conceptual soundness asks whether the modelling approach is appropriate to the problem, whether the data supports the inference being drawn, and whether the assumptions are stated and defensible. For language model applications this question has proved the most difficult to answer rigorously, because the approach is rarely selected on principled grounds and is more often selected because it worked.

Implementation verification requires reproduction. The validator regenerates the headline metrics from the registered data and the submitted code on validation-controlled infrastructure. Divergence beyond a stated tolerance is a finding regardless of direction, and a model that performs better on reproduction than on submission is investigated with the same seriousness as one that performs worse.

Outcome analysis tests performance on data the model has not seen, including data from periods after the training window where available. Validators construct their own test sets in addition to reviewing the team's, and they are specifically instructed to test regions the team's evaluation does not cover, because the team's blind spots are the point of independent review.

## Challenger models

For H1 and H2 models the validator constructs or commissions a challenger, meaning an independent model built to address the same problem by different means. The challenger's purpose is not to be better. It is to establish what performance is achievable, so that the champion's performance can be interpreted rather than merely recorded.

A champion that substantially outperforms a competent challenger is evidence that the modelling approach adds value. A champion that matches a far simpler challenger raises the question of whether the complexity is justified, and Meridian has retired several sophisticated models on precisely this finding. A champion that underperforms its challenger is a finding that blocks approval.

Challengers are also the firm's principal defence against a specific failure of judgement: the tendency to accept a model's performance as adequate because it is the only performance anyone has measured. Without a reference point, a validator assessing whether a metric is good enough is making an unanchored judgement, and unanchored judgements drift toward whatever the team presents.

For language model applications the challenger is frequently a substantially simpler configuration: the same base model with better prompting and no fine-tune, or retrieval with a smaller model, or in several documented cases a deterministic template. The firm's experience is that a meaningful minority of proposed generative applications are outperformed by their own challenger, and the discipline of building one has saved more engineering effort than it has consumed.

## Benchmarking the fine-tune

A fine-tuned model is validated against three references rather than one. The first is the base model with no adaptation, which establishes what the fine-tune contributed. The second is the base model with prompting alone, which establishes whether the fine-tune was necessary. The third is the previous production version where one exists, which establishes whether the change is an improvement.

Reporting only the first comparison is the most common shortcoming in fine-tuning submissions. A team demonstrating that their fine-tuned model beats the raw base model has demonstrated that training did something, which was never in doubt. The question the validator asks is whether it did something that prompting could not have done more cheaply and more reversibly.

Validators are instructed to run the prompted-baseline comparison themselves rather than accept the team's version, because the quality of the baseline prompt determines the outcome and the team constructing it has an interest. A weak baseline prompt is the easiest way to make a fine-tune look valuable, and it is usually not deliberate.

## Findings and their disposition

Findings are graded as material, significant, or observational. A material finding blocks approval until remediated. A significant finding may be accepted with a remediation plan and a deadline. An observational finding is recorded and tracked but imposes no obligation.

Grading is the validator's decision and is not negotiable with the model owner, though the owner may appeal to the model risk committee. Appeals are uncommon and are resolved in the validator's favour in most cases, which the firm regards as evidence that the grading is calibrated rather than that the process is captured.

Accepted findings are tracked to closure and reviewed at the next revalidation. A finding accepted with a remediation plan that was not executed becomes material at revalidation, which prevents the accumulation of permanently deferred obligations. This ratchet was introduced after a review found several models carrying accepted findings that were four years old.

## Validating what cannot be reproduced

Third-party models present a structural difficulty: the validator cannot inspect the training data, cannot reproduce the training, and cannot rule out that the vendor's evaluation overlaps the vendor's training. Meridian's response is to shift the validation emphasis from construction to behaviour, testing the model extensively on the firm's own data and treating vendor claims as unverified assertions.

Where a vendor model is a component of a firm-built system, validation addresses the system. The validator establishes how the system behaves when the component behaves badly, which is a more useful question than whether the component is good, because the component will change without notice and the system's tolerance for that change is the property the firm controls.

Vendor model changes are monitored through continuous evaluation, and a detected behavioural change triggers a validation review rather than merely a finding. The firm treats an unannounced vendor model change as equivalent to an unauthorised production change, and the contractual arrangements the firm now negotiates require notice and version pinning specifically because this control was otherwise impossible to operate.

## The validator's report

A validation report states the scope of what was reviewed, the tests performed, the findings with their grades, the validator's opinion on residual risk, and any conditions attached to approval. It also states what was not reviewed and why, which is the section most often omitted and most useful to a supervisor.

Reports are written for a reader three years in the future with no context and no access to anyone involved. This standard is applied literally during quality review of the validation function's own work, and reports that assume shared context are returned. The function's institutional memory lives in these documents, and a report that requires its author to interpret it has failed at its primary purpose.
