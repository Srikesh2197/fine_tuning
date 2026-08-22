# Lattice: The Language Model Serving Platform

## What Lattice is

Lattice is Meridian Trust's internal gateway for all large language model inference. Every request from every application, whether destined for a self-hosted open-weight model or a third-party hosted endpoint, transits Lattice. No application at the firm holds a vendor API key directly, and the network policy enforces this by blocking egress to known inference providers from application subnets.

Centralising inference was not primarily a cost decision, though it produced cost benefits. It was a control decision. A firm with fifty teams calling inference providers independently has fifty places where client data might leave the perimeter, fifty prompt-injection surfaces, and no coherent answer to a supervisor asking what the firm sends to which vendor. Lattice reduces that to one enforcement point.

## Request lifecycle

A request arriving at Lattice carries an application identity, an acting user identity, and a registered model alias. Lattice resolves the alias to a concrete backend, checks that the calling application is authorised for that alias, checks that the request falls within the model's declared operating envelope, applies the configured Scrim redaction policy, forwards the request, and records the interaction in the audit log before returning the response.

The acting user identity is mandatory and is not the same as the application identity. Applications that cannot supply a user identity may only call aliases marked as non-attributable, which excludes every alias with access to client data. This constraint caused significant friction during rollout and has since prevented several incidents in which a batch process would otherwise have queried client information without any user's authority behind it.

Lattice enforces the operating envelope declared in each model's Blue-file. Requests exceeding the declared maximum input length, arriving at a rate above the declared throughput, or originating from a region inconsistent with the model's data residency constraint are rejected with a specific error rather than silently truncated or routed elsewhere. Silent accommodation of out-of-envelope requests was an early design mistake and produced a class of failures nobody could reproduce.

## Aliases and the registry

Applications call model aliases rather than model names. An alias such as `research-summariser-v3` resolves to a specific backend, a specific prompt template version, a specific decoding configuration, and a specific redaction policy. The indirection allows the platform team to migrate the underlying model without touching application code, and it allows governance to attach approval scope to something stable.

Alias resolution is versioned and auditable. Every response returned by Lattice carries a resolution stamp recording the alias, the concrete backend, the template version, and the adapter identifier if one was applied. Reconstructing what actually produced a given output six months later is therefore a lookup rather than an investigation, which matters enormously during validation and during incidents.

The registry rejects aliases whose backing model is not registered in Sable with a current Gatepost 3 approval. This is the mechanism by which governance is made structural rather than procedural. A team that skips Gatepost cannot deploy, not because a policy forbids it but because the platform will not resolve their alias.

## Routing and model selection

Lattice routes on capability class rather than on vendor. Applications declare the class of work they need, and the platform maps that class to an appropriate backend. Classes in current use include long-context extraction, structured output generation, conversational assistance, embedding, and reranking. Each class has a primary backend and at least one fallback of comparable capability.

Fallback is automatic on availability failure and manual on quality failure. Availability failures are unambiguous and the platform handles them within the request. Quality failures are contested, slow to establish, and frequently application-specific, so switching primary backends for quality reasons requires evidence from Plumb and a delegated Gatepost approval.

The platform deliberately does not expose a raw passthrough to vendor endpoints. Teams periodically request one, usually to access a capability the classes do not yet cover, and the answer is to extend the class taxonomy rather than to create an ungoverned path. Every passthrough that has been granted historically has become permanent, and the firm now treats such requests as a signal that the taxonomy needs work.

## Quotas, cost, and capacity

Every application holds a token budget denominated in cost rather than in tokens, refreshed monthly, and attributed to the owning cost centre. Denominating in cost rather than tokens was a late change and a correct one, because token-denominated quotas created incentives to route to cheap models regardless of fitness, and produced applications that were technically within quota while being economically absurd.

Lattice publishes consumption per application, per alias, and per acting user. Per-user visibility exists for capacity planning and anomaly detection, not for individual performance management, and this boundary is stated explicitly in the platform's own documentation because the ambiguity was raised repeatedly during rollout.

Capacity for self-hosted models is managed as a shared pool with per-class reservations. Interactive classes hold guaranteed capacity; batch classes consume the remainder and are preempted under load. Batch workloads that cannot tolerate preemption must declare so and are charged at a substantially higher rate, which has proved an effective way of establishing which workloads are genuinely time-critical.

## Prompt injection and untrusted content

Lattice treats all retrieved and user-supplied content as untrusted. The platform provides a structural separation between instruction content, which originates from the registered prompt template, and data content, which originates from retrieval or from the user. Applications that concatenate the two themselves before calling Lattice are in breach of platform policy, and the request schema is designed to make the correct pattern the easy one.

Structural separation reduces prompt injection risk but does not eliminate it, and the platform's documentation says so plainly. The residual control is that model outputs are never granted authority they did not already have. A model output may not trigger a payment, may not modify an entitlement, and may not initiate an outbound communication without a human action that is itself authenticated and authorised.

This principle, that model output is a recommendation and never an authority, is the single most important security property of the platform. It means that a successful injection produces bad advice rather than a bad transaction. Applications requesting an exception are directed to reframe the workflow so that the consequential step is taken by a person.

## Observability

Lattice records every request and response, the resolution stamp, the redaction actions Scrim applied, latency at each stage, token counts, and the outcome. Prompt and response bodies are retained under a shorter retention period than the metadata, and access to bodies requires a specific entitlement that is granted to validation, to the platform team for incident response, and to nobody else by default.

The audit log is the substrate for most of the firm's language model governance. Revalidation samples from it, drift monitoring computes DXI from it, incident investigation reconstructs from it, and capacity planning aggregates it. Teams building on Lattice therefore inherit a large fraction of their monitoring obligations by construction, which is the strongest argument the platform team makes when a team proposes building their own serving path.
