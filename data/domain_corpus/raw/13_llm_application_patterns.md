# Application Patterns for Language Models

## The pattern catalogue

Meridian Trust maintains a catalogue of approved application patterns for language model systems. A team proposing a new application is expected to identify which pattern it follows, and a proposal that fits no pattern is not prohibited but attracts substantially more design scrutiny at Gatepost 1. The catalogue exists because the same architectural mistakes were being made independently by teams who had no way to learn from each other.

Five patterns account for the overwhelming majority of production systems: bounded extraction, grounded question answering, drafting with review, classification and routing, and supervised multi-step workflow. Each has a characteristic risk profile, a characteristic evaluation approach, and a characteristic failure mode, and knowing which pattern applies tells a reviewer most of what they need to ask.

## Bounded extraction

Bounded extraction takes a document and produces a structured record with a fixed schema. Extracting counterparty names and notional amounts from a confirmation, pulling obligations from a contract, or normalising a corporate action notice are all bounded extraction. The output space is constrained, which makes the pattern the easiest to evaluate and the safest to deploy.

The characteristic failure is confident extraction of a field that is absent. A model asked for a maturity date on a document that does not state one will frequently supply a plausible date rather than a null, and the downstream system has no way to distinguish it. Extraction schemas at Meridian therefore require an explicit absent marker and a confidence indication per field, and evaluation measures null accuracy separately from value accuracy.

Bounded extraction is the pattern where fine-tuning most reliably pays for itself. A small model fine-tuned on a few thousand examples of the firm's own document types typically matches a much larger prompted model at a fraction of the cost and latency, and the output schema stability improves markedly. Several of the firm's highest-volume language model workloads are fine-tuned extraction models.

## Grounded question answering

Grounded question answering retrieves relevant passages and answers from them. It is the most requested pattern and the most frequently misjudged, because a demonstration is easy and a production system is not. The gap between the two is almost entirely about the questions the demonstration did not include: out-of-corpus questions, questions with false premises, questions requiring synthesis across contradictory sources, and questions whose answer depends on an effective date.

The characteristic failure is unsupported assertion, in which the model supplies from parameters what retrieval did not supply from the corpus. This is why abstention behaviour is evaluated explicitly, why the adversarial share of a Plumb suite is set at forty per cent, and why Quarry returns an empty result rather than weak passages.

Grounded question answering is the pattern where fine-tuning most often disappoints. Teams fine-tune on their corpus expecting better answers and get a model that is more confident and less grounded. Domain adaptation helps this pattern by improving the model's handling of firm vocabulary, which improves both retrieval query understanding and answer fluency, but it does not substitute for retrieval and should not be attempted as one.

## Drafting with review

Drafting with review produces a first draft that a qualified person edits and owns. Client correspondence, credit memoranda, meeting notes, and internal papers all fit. The pattern's safety rests entirely on the reviewer's engagement, which degrades predictably as draft quality improves, and this is the pattern's central and unresolved tension.

The characteristic failure is automation complacency. A reviewer editing drafts that are correct ninety-five per cent of the time stops reading carefully by the third week. Meridian's mitigations are to require the model to surface uncertainty explicitly rather than smooth it away, to require citation of source material within drafts, and to sample reviewed outputs for independent quality assessment rather than trusting that review occurred.

Drafting systems must never produce output that could be sent without a review action. Applications that pre-populate a send field, that default to approve, or that make review a single keystroke are rejected at design review, because the control being relied upon is the reviewer's attention and the interface should cost attention rather than save it.

## Classification and routing

Classification and routing assigns an input to a category that determines downstream handling: which queue an enquiry enters, which team receives an alert, which template applies. The output is a small discrete set, which makes evaluation straightforward and makes conventional classification metrics directly applicable.

The characteristic failure is silent miscategorisation into a low-attention queue. An item routed to a queue nobody watches closely disappears, and the failure is invisible in aggregate accuracy because it is a small fraction of volume concentrated in a category that matters. Evaluation is therefore per-class with declared floors, and routing systems are required to monitor per-queue volume for distributional shifts.

Classification is the pattern most often better served by a smaller model than a language model, and the challenger for a proposed classification application is frequently a fine-tuned encoder or a gradient-boosted model over engineered features. Validators are specifically instructed to require this comparison, and it changes the decision more often than teams anticipate.

## Supervised multi-step workflow

Supervised multi-step workflow chains several model calls with deterministic steps between them, under a defined process with a human decision point before any consequential action. Reconciliation investigation, client onboarding document review, and control testing all follow this pattern.

The characteristic failure is error compounding. A step operating at ninety-five per cent accuracy is acceptable; five such steps in sequence are not, and the arithmetic surprises teams who evaluated each step independently. Multi-step workflows must be evaluated end to end on the full chain, and the Plumb suite for such a system evaluates the chain rather than the components, with component evaluation retained for diagnosis only.

The second failure is state accumulation, in which an early error propagates into context for later steps and is thereafter treated as established fact. Meridian's guidance is to design steps so that each can be checked against source material rather than against the previous step's output, and to insert deterministic validation between steps wherever the intermediate result has a checkable structure.

## Patterns the firm does not permit

Autonomous action without a human decision point is not an approved pattern for any system whose action has external effect. This is not a statement about model capability. It is a statement about the firm's ability to explain, to a customer or a supervisor, why something happened, and about the availability of a person accountable for the decision.

Open-ended conversational agents exposed directly to customers are not currently approved. The firm operates several such systems internally, and the difference is that an internal user is an employee subject to policy who can be told what the system is for, while an external user is not. Proposals in this space are directed toward the grounded question answering pattern with a constrained scope.

Systems that modify their own prompts, tools, or retrieval configuration at runtime are not permitted. The registered configuration is what was approved, and a system that changes it has escaped its approval. Adaptive behaviour is achievable within an approved configuration space, and applications requiring it must declare the space at Gatepost rather than the mechanism for exploring it.
