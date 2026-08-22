# The Prompt Engineering Standard

## Prompts are code

Meridian Trust treats prompt templates as model code rather than as configuration. They are versioned in the application repository, they are reviewed at merge, they appear in full in the Blue-file, and changing one is a material change that resets the revalidation clock. This position was contested when it was introduced and is now uncontroversial, because the intervening experience made the argument better than any policy paper could.

The argument is simple. A prompt template determines the model's behaviour at least as strongly as any hyperparameter, and frequently more strongly. A firm that governs its training configuration rigorously and its prompts not at all has governed the less important half. Ungoverned prompts also decay, because an untracked change made under time pressure is a change nobody can find later.

## Template structure

An approved template has four zones in a fixed order: role and task definition, operating constraints, provided context, and the request. The zones are structurally delimited rather than merely separated by prose, and the platform's request schema carries them as distinct fields so that Lattice can enforce the separation between instruction content and data content.

The role and task zone states what the system is and what it must produce. It is written in specific rather than aspirational terms, because instructions to be helpful and accurate do not constrain behaviour while instructions to answer only from the provided passages do. Templates containing aspirational instruction are returned at review with a request to state the operative constraint instead.

The operating constraints zone states what the system must not do, what it must do when it cannot comply, and what format the output must take. The middle of these is the one teams omit and the one that determines production behaviour, because the interesting cases are precisely the ones where compliance is impossible and the model must choose a failure mode.

The provided context zone carries retrieved passages, user-supplied documents, and any structured data the task requires. Content in this zone is untrusted by construction, and templates are required to state explicitly that instructions appearing within it are to be treated as data. This instruction is not a complete defence against prompt injection and the standard says so, but it measurably reduces the success rate of naive attempts.

The request zone carries the user's actual question or instruction. It is last because recency helps, and it is separate because conflating it with the task definition is how applications end up granting users the ability to redefine the system's purpose.

## Structured output

Applications requiring structured output declare a schema and receive validated output or an error, never unvalidated text that the application then parses hopefully. Lattice supports constrained decoding where the backend does, and schema validation with bounded retry where it does not.

Schemas should be as constrained as the task allows. Enumerations are preferable to free strings, nullable fields must be genuinely nullable rather than represented by a sentinel string, and nested structures should be flattened where the nesting carries no meaning. Every degree of freedom in a schema is a degree of freedom in the failure modes.

The retry policy on validation failure is bounded at two attempts, after which the request fails. Unbounded retry converts a quality problem into a cost problem and conceals the underlying failure rate, and the platform reports validation failure rates per alias precisely so that the underlying rate stays visible.

## Few-shot examples

Few-shot examples are permitted and are subject to the same versioning as the rest of the template. They are counted toward the token budget and are reviewed for the same sensitivity concerns as any other content, since examples drawn from real cases are a recurring source of inadvertent disclosure.

Examples must be drawn from the same distribution as production inputs, and examples constructed by hand to illustrate a point are a known source of degradation. A model shown three unusually clean examples learns that inputs are clean, and its behaviour on the messy majority worsens. Where examples are constructed rather than sampled, the Blue-file must say so.

Where an application requires more than roughly eight examples to perform acceptably, the standard directs the team to consider fine-tuning instead. Long example blocks consume context that retrieval could use, they cost tokens on every request, and the behaviour they install is more reliably installed by training. This threshold is guidance rather than a rule, and the reasoning matters more than the number.

## Instructions that do not work

The standard maintains a list of instruction patterns that testing has shown to be ineffective, so that teams stop reinventing them. Instructing the model not to hallucinate does not reduce unsupported assertions measurably. Instructing it to be concise without stating a length produces variable output. Instructing it to think carefully produces no reliable improvement on the firm's tasks when the model already reasons before answering.

Instructing the model to cite sources works only when the citation format is specified precisely and the provided context carries stable identifiers. Vague citation instructions produce citations that look right and point nowhere, which is worse than no citation because users trust them.

Negative instructions are less reliable than positive ones. Telling the model what to do instead of what to avoid produces more consistent behaviour, and templates that consist largely of prohibitions are returned at review with a request to restate them as requirements.

## Template review

Templates are reviewed by a second engineer at merge and by validation at Gatepost 2. The engineering review checks structure, zone separation, and the absence of sensitive content. The validation review checks that the template's constraints correspond to the behaviour the Blue-file claims, which is a check teams find surprisingly difficult to pass on first submission.

The most common validation finding is that the Blue-file describes a behaviour the template never asks for. A limitations section stating that the system declines out-of-scope questions, paired with a template that gives the model no instruction about scope, describes an aspiration rather than a system. The finding is graded significant, because the documentation is wrong rather than merely incomplete.

Templates are re-reviewed at every revalidation against the current adversarial suite, and templates that have not changed while the threat picture has are treated as a finding rather than as evidence of stability.
