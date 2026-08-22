# Tool Use and Supervised Agentic Workflows

## Authority is the governing concept

Meridian Trust permits language models to invoke tools under a single organising principle: a model output is a recommendation and never an authority. A tool a model may invoke is a tool whose effects the firm is willing to accept from an untrusted source, because the model's input includes retrieved content and user content, both of which an adversary may influence.

This principle resolves most design questions before they become debates. A model may query a position; it may not move one. A model may draft a message; it may not send one. A model may propose a limit change; the change is effected by a person acting on their own authority, with the model's proposal visible as an input to their decision rather than as an instruction to be confirmed.

The distinction between acting on a proposal and confirming an instruction is behavioural rather than technical, and interface design carries the burden. A screen presenting a proposal alongside the evidence for it, requiring the reviewer to select an action, produces a decision. A screen presenting a filled form with a confirm button produces a keystroke.

## Tool classification

Tools are classified as read, compute, or effect. Read tools retrieve information and change nothing. Compute tools transform data and produce results without persisting them. Effect tools change state visible outside the workflow, whether by writing to a system of record, sending a communication, or initiating a process.

Read and compute tools may be invoked by a model directly, subject to entitlements evaluated against the acting user exactly as Quarry evaluates them. Effect tools may not be invoked by a model under any circumstances. An effect tool is invoked by the application, following a human decision, with the model's proposal recorded as an input to that decision.

Read tools are not risk-free and the classification does not suggest otherwise. A read tool with broad access allows an injected instruction to exfiltrate data through the model's response, and the mitigations are entitlement scoping, output inspection through Scrim, and the general constraint that the response returns to the acting user rather than to an arbitrary destination.

## Scoping and least privilege

Every tool registration declares the entitlements it requires and the data classes it can reach. Lattice evaluates the intersection of the tool's declared scope, the application's registered scope, and the acting user's live entitlements, and the effective scope is the intersection rather than any one of them.

Tools that require broad access to function are redesigned rather than approved. A tool that must read every position to answer a question about one is a query design problem, and the platform team works with the application team to narrow it. Broad-scope tools have been approved on a handful of occasions, always with the consuming application classified at H1 regardless of its apparent purpose.

Tool schemas are versioned and registered like prompt templates. A tool whose parameters change has changed the model's action space, which is a material change. Applications discovering that a tool's behaviour changed underneath them is a failure mode the registry exists to prevent.

## Loop control

Multi-step workflows run under explicit step limits, wall-clock limits, and cost limits, all declared at registration and enforced by the platform. A workflow that exhausts a limit terminates with a recorded reason and surfaces its partial state, rather than continuing or failing silently.

Step limits are set from the workflow's designed depth plus a small margin, not from a generous default. A workflow designed to complete in four steps and configured with a limit of forty will, when something goes wrong, spend thirty-six steps going wrong expensively. The platform reports the distribution of steps-to-completion so that limits can be set from evidence.

Repetition detection terminates workflows that invoke the same tool with the same arguments beyond a threshold. This is the most common pathological loop and the cheapest to detect, and it accounts for the majority of limit terminations in production. The termination is recorded as a defect signal rather than as a normal outcome, and applications exceeding a threshold rate are reviewed.

## Context accumulation

A multi-step workflow accumulates context, and accumulated context is where errors become facts. A tool result misinterpreted at step two is carried into steps three through six as established input, and no later step has any way to question it.

The firm's guidance is to structure workflows so that each step can be validated against source material rather than against the previous step's output, and to insert deterministic validation between steps wherever the intermediate result has checkable structure. An extracted identifier can be checked against a system of record; an extracted judgement cannot, and workflows that chain judgements are inherently fragile.

The second guidance is to keep raw tool output in context rather than the model's summary of it, where token budget allows. Summarisation between steps is lossy in ways that correlate with what the model considered unimportant, which is precisely the information a later step may need.

## Evaluation of workflows

Workflows are evaluated end to end, on the full chain, against cases that include tool failure, tool timeout, empty results, and contradictory results between tools. Component-level evaluation is retained for diagnosis and is not sufficient for release, because the compounding of per-step error rates is the property that determines whether the workflow is usable.

The adversarial share of a workflow's Plumb suite includes injection attempts placed in tool results rather than only in user input. A tool returning a document containing instructions is the realistic attack, and a suite that tests injection only through the user-facing input has tested the easier half.

Workflows are also evaluated on their behaviour at limits. A workflow that terminates on step limit should surface a useful partial result and a clear statement of what it did not complete, and the suite includes cases constructed to hit each limit so that the degraded path is exercised rather than assumed.

## Human decision points

Every workflow with an effect step has a designated human decision point, and the Blue-file names the role that occupies it and the information that role receives. The information matters as much as the role: a decision point where the reviewer sees a recommendation without the evidence is a rubber stamp, and validation tests the sufficiency of the presented information rather than the presence of the step.

Decision points are placed before irreversible effects rather than after the workflow completes. A workflow that runs to completion and then asks for approval of everything it did has placed the decision where it cannot be acted on meaningfully, and such designs are returned at Gatepost 1.

Where a workflow's volume makes per-item human decision impractical, the correct response is to narrow the workflow's scope rather than to weaken the decision point. The firm has declined several proposals on this basis, and the resulting narrower systems have generally delivered most of the intended value with a control model that holds.
