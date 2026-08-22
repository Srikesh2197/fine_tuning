# Scrim: Sensitive Data Detection and Redaction

## Purpose

Scrim is the service that inspects and transforms data crossing trust boundaries at Meridian Trust. Its principal deployment is inline within Lattice, where it inspects every prompt before it reaches an inference backend and every response before it returns to the calling application. It also runs at index time within Quarry and at ingestion within Ledgerline.

Scrim is a control, not a safety net. The firm's position is that applications must not send data they are not entitled to send, and Scrim exists to catch the cases where an application is wrong about what it holds. Teams that architect around Scrim, sending unfiltered data on the assumption that redaction will handle it, are operating outside policy even when the outcome happens to be acceptable.

## What Scrim detects

Scrim detects direct identifiers, quasi-identifiers, and firm-confidential markers. Direct identifiers include names, account numbers, national identifiers, payment card numbers, contact details, and device identifiers. Quasi-identifiers include date and place of birth, employer, precise geographic references, and rare attribute combinations that are individually innocuous and jointly identifying.

Firm-confidential markers cover material non-public information, unpublished research, deal codenames, and unreleased financial data. Detection here is largely lexicon-driven and list-maintained rather than model-driven, because the categories are enumerable and the cost of a miss is high enough that recall matters far more than elegance.

Detection combines pattern matching with checksum validation for structured identifiers, a named-entity model for unstructured references, and corpus-specific lexicons. Checksum validation matters more than it sounds: a sixteen-digit number that fails the Luhn check is usually a reference number rather than a card, and validating it reduces the false positive rate enough to keep the service usable.

## Transformation modes

Scrim supports four transformation modes: block, redact, tokenise, and pass with annotation. The mode is configured per corpus, per alias, and per detected category, which produces a policy matrix that is more complex than teams would like and less complex than the problem.

Block rejects the request outright. It is used for categories that must never leave a boundary, such as full payment card numbers reaching any external inference provider. Blocking is unambiguous and produces a clear error, which is preferable to a silent transformation that leaves the application unaware anything happened.

Redact replaces the detected span with a category placeholder such as a bracketed account-number marker. It is used where the model does not need the value to do its work, which is more often than teams initially believe. A summarisation task rarely needs the account number, and removing it costs nothing.

Tokenise replaces the value with a stable surrogate that can be reversed by an authorised service on the return path. Tokenisation is what makes personalised generation possible without exposing identifiers to the inference backend, and it is the mode most applications end up using. The surrogate is stable within a session and unstable across sessions, so that a backend cannot accumulate a profile across interactions.

Pass with annotation permits the value through and records that it was permitted. This mode exists for self-hosted models running within the firm's own perimeter, where the data has not in fact crossed a trust boundary. Using it for an external backend requires a documented exception approved at Gatepost, and such exceptions are rare and time-limited.

## Tokenisation in detail

The tokenisation surrogate preserves format class but not value. An account number becomes a syntactically valid account number that belongs to no account; a name becomes a plausible name of similar length and structure. Format preservation matters because models behave differently when the input is malformed, and a redaction that turns a name into a placeholder changes the task in ways that degrade output quality measurably.

The mapping is held in a session-scoped vault with a short lifetime, is never written to durable storage, and is inaccessible to the inference path. Reversal happens on the return path inside Lattice, after the response leaves the backend and before it reaches the application. The application therefore sees real values, the backend never does, and neither is aware of the substitution.

Tokenisation has a failure mode that teams should understand: a model asked to reason about a tokenised value may reason about the surrogate. Asked whether two accounts belong to the same customer, a model sees two surrogates and can only compare them as strings. Applications requiring reasoning over identifier semantics must either use a self-hosted backend or restructure the task so that the comparison happens outside the model.

## Data residency

Scrim enforces residency alongside detection. Each corpus and each acting user carries a residency constraint, and Lattice will not route a request to a backend outside the permitted region. Residency is evaluated on the union of constraints attached to the request, so a request combining data from two regions is restricted to backends permitted for both.

The practical consequence is that certain capabilities are unavailable in certain regions, and applications must handle this rather than assume global capability parity. The platform surfaces the available capability set per residency context so that applications can degrade deliberately, and Plumb suites for multi-region applications are required to evaluate the degraded path as well as the primary one.

Residency constraints are not negotiable through the exception process. Where a capability is genuinely required and cannot be served in-region, the answer is to bring the capability in-region, which the firm has done several times at considerable expense, rather than to route the data.

## Performance and false positives

Scrim adds latency, and the latency is visible. Inline detection on a long prompt costs tens of milliseconds, which matters for interactive applications and is negligible for batch. The service is optimised for the interactive path, and detection on long documents at index time runs on a separate throughput-optimised deployment.

False positives are the dominant operational complaint. A lexicon entry for a deal codename that happens to be a common word produces redaction across unrelated content, and users experience this as the system mangling their text. Lexicon entries are therefore reviewed for collision before activation, and the service reports per-entry hit rates so that pathological entries surface quickly.

False negatives are measured through periodic red-team exercises rather than through production monitoring, because a missed identifier that reaches a backend leaves no signal in the normal telemetry. These exercises run quarterly against a constructed corpus with known planted identifiers, and the results feed the detection roadmap.

## Interaction with fine-tuning

Training data receives the same treatment as inference data, with one important difference: tokenisation is unavailable because there is no return path. Training corpora must therefore be redacted or must contain only data the model is permitted to memorise, and the firm's default is that no model is permitted to memorise client identifiers.

This constrains what can be fine-tuned on. A domain adaptation corpus assembled from internal policy and procedure documents is straightforward, because such documents rarely contain client identifiers and Scrim confirms it. A corpus assembled from client correspondence is not, and teams proposing it are directed toward retrieval, where entitlements are enforced per request and nothing is memorised.

The distinction between memorisation and retrieval is the organising principle for the whole question of what may be fine-tuned on. Retrieval keeps sensitive content behind an access check that is evaluated at request time. Fine-tuning bakes content into weights that are then served to everyone with access to the model, and no access check exists inside a weight matrix.
