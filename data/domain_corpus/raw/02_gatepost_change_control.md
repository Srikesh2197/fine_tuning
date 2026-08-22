# Gatepost: Change Control for Models

## What Gatepost is

Gatepost is the change-control process every model at Meridian Trust passes through on its way to production. It consists of three sequential gates, numbered Gatepost 1, Gatepost 2, and Gatepost 3. The gates are not review meetings, although meetings occur within them. Each gate is a decision point at which a named approver either permits the model to advance or returns it with recorded findings, and the decision is recorded in Sable against the model entry.

Gatepost 1 is the design gate. It occurs before substantial engineering effort has been spent, and it exists to prevent the firm from building things it will not be permitted to deploy. Gatepost 2 is the pre-validation gate, at which the model is functionally complete and the validation function formally accepts it into the validation queue. Gatepost 3 is the production release gate, at which the model is authorised to receive production traffic.

## Which gates apply to which tiers

H1 and H2 models must pass all three gates. H3 models must pass Gatepost 3 only, on the reasoning that design-stage review of internal productivity tooling costs more than it returns. H4 models are exempt from Gatepost entirely, which is the substantive privilege of sandbox status and the reason teams are willing to accept the constraint that H4 work may not touch production data.

A model that changes tier mid-development does not retroactively acquire gates it has already passed the equivalent stage for, but it does acquire all subsequent ones. A tool that begins as an H3 productivity aid and is reclassified to H2 after a scope expansion must complete Gatepost 2 before proceeding, even though Gatepost 1 is no longer meaningfully achievable. In practice the validation function treats this as a combined Gatepost 1 and 2 review and records it as such.

## Gatepost 1: design

The Gatepost 1 submission is deliberately short. It states the business problem, the proposed modelling approach, the intended Halton tier with reasoning, the data the model will consume and the legal basis for consuming it, the population the model will be applied to, and the decision the output will inform. It also states what the team will accept as evidence that the model works, which is the single most useful sentence in the document and the one teams find hardest to write.

The approver at Gatepost 1 is the divisional model risk lead. Approval is not an endorsement of the approach and carries no weight at later gates. It confirms only that the proposed use is permissible, that the data basis is sound, and that the tier assignment is plausible. Roughly one in five Gatepost 1 submissions is returned, and the overwhelming majority of returns concern the data basis rather than the modelling approach.

The most valuable outcome of Gatepost 1 is negative. A team told at design stage that the data they intended to use cannot lawfully be used for this purpose has lost a week. The same team told at Gatepost 3 has lost a quarter, and will have built an evaluation suite, a serving path, and a monitoring dashboard around data it must now discard.

## Gatepost 2: pre-validation

Gatepost 2 is where the model is handed to independent validation. The submission comprises the completed Blue-file, the training and evaluation datasets registered in Ledgerline with immutable version identifiers, the full evaluation results from Plumb, the model artifacts themselves, and a reproduction script that allows the validator to regenerate the headline metrics from scratch.

The reproduction requirement is enforced literally. The validator runs the script on validation-controlled infrastructure, and if the headline metrics do not reproduce within a stated tolerance the submission is returned without further review. This single control has done more for the quality of the firm's model estate than any other, because it makes undisclosed manual steps impossible to sustain.

Validation acceptance at Gatepost 2 does not mean the model is approved. It means the submission is complete enough to review. The distinction matters for planning, because teams that treat Gatepost 2 acceptance as a milestone toward release consistently underestimate the validation period that follows, which for an H1 model averages seven weeks.

## Gatepost 3: production release

Gatepost 3 authorises production traffic. The submission adds the operational material that did not exist at earlier gates: the deployment topology, the serving region and its data residency implications, the rollback procedure with a demonstrated rollback time, the monitoring configuration including DXI thresholds, the amber-window sampling plan, and the on-call rotation that will respond to alerts.

The approver at Gatepost 3 is the Gatepost board, which convenes weekly and comprises the divisional model risk lead, a representative of the validation function, the platform owner for the serving environment, and for H1 models a representative of compliance. The board does not re-litigate validation findings. It confirms that findings have been closed or formally accepted, and that the model can be operated and withdrawn safely.

A Gatepost 3 approval is scoped. It authorises a specific model version, serving a specific population, from a specific region, under a specific monitoring configuration. Changing any of those requires either a fresh Gatepost 3 or, for narrowly defined changes, a delegated approval recorded against the original. Teams frequently assume that approval attaches to the model rather than to the deployment, and the platform enforces the correct interpretation by rejecting configuration that does not match the approved scope.

## Delegated and expedited paths

Not every change warrants a full board review, and a process that pretends otherwise is a process teams route around. Meridian therefore operates a delegated path for changes that do not alter the model's decision behaviour: infrastructure migration within an approved region, dependency patching, observability changes, and capacity adjustment. The divisional model risk lead approves these unilaterally and reports them to the board retrospectively.

An expedited path exists for defect remediation. If a production model is behaving incorrectly and the fix is understood, the on-call model risk lead may authorise release with a single approval, provided the change is reviewed by the full board within five business days. The expedited path is used perhaps a dozen times a year across the firm, and every use is reported to the model risk committee.

The expedited path may not be used to accelerate a change that is merely commercially urgent. This boundary is policed carefully, because the failure mode is obvious and the pressure is real. The test applied is whether the current production state is causing harm. Commercial opportunity cost is not harm for this purpose.

## What Gatepost does not do

Gatepost does not assess whether a model is good. That is validation's role, and conflating the two produces a board that debates hyperparameters and a validation function that assumes the board has covered the fundamentals. Gatepost assesses whether the model is permitted, documented, reproducible, monitored, and reversible.

Gatepost also does not manage the model after release. Once a model is live, accountability sits with the model owner and the monitoring regime, and the next Gatepost interaction occurs only at the next material change or at the revalidation triggered by the Halton cadence. Teams that treat Gatepost 3 as the end of the process are the teams whose models are found to have drifted at revalidation.

## Records and evidence

Every Gatepost decision produces a durable record comprising the submission as presented, the findings raised, the approver identity, the decision, and any conditions attached. These records are retained for the life of the model plus seven years and are the primary evidence the firm produces when a supervisor asks how a particular model came to be in production.

The quality of that evidence is a control objective in its own right. A record that says a model was approved tells a supervisor nothing. A record that says which alternatives were considered, which findings were raised, which were closed and how, and which were accepted with reasoning demonstrates that a decision was made rather than a form completed. Teams are coached to write submissions that will read well to someone reconstructing the decision three years later without access to anyone who was present.
