# Quarry: Entitlements-Aware Retrieval

## Purpose

Quarry is Meridian Trust's retrieval service for language model applications. It indexes internal document corpora and returns passages relevant to a query, filtered by the entitlements of the acting user. The entitlements filtering is the defining feature and the reason the firm built its own service rather than adopting a general-purpose vector database directly.

A retrieval system inside a bank cannot treat access control as a post-processing step. Filtering results after retrieval leaks information through the shape of the result set, through relevance scores, and through latency. Quarry therefore applies entitlements as a pre-filter on the candidate set, so that a passage the user may not see is never a candidate and never influences ranking.

## Index structure

Quarry maintains separate indices per corpus, where a corpus is defined by a shared entitlement model and a shared refresh cadence rather than by subject matter. Policy documents, research notes, client correspondence, and operational runbooks are distinct corpora because their access rules and update rhythms differ, even where their subject matter overlaps.

Each indexed passage carries the passage text, a dense embedding, a sparse lexical representation, the source document identifier, the position within the source, the effective date range, and the entitlement tuple required to retrieve it. The entitlement tuple is resolved at index time from the source system's own access model, and Quarry never infers entitlements from document content.

Effective date ranges matter more in a bank than in most retrieval settings. A policy superseded last quarter is not merely stale, it is wrong, and answering from it is a compliance failure rather than a quality problem. Quarry supports point-in-time retrieval so that an application asking what the policy was on a given date receives the passages that were effective then, which is essential for anything supporting a dispute or an audit.

## Chunking

Quarry chunks on structural boundaries first and falls back to token windows only where structure is absent. Headings, list boundaries, table boundaries, and clause numbering are all preferred split points, because a passage that begins mid-sentence or spans two unrelated clauses retrieves poorly and reads worse when surfaced to a user.

The default target is four hundred tokens per chunk with an overlap of sixty tokens, but the defaults are overridden per corpus and frequently should be. Dense policy text with numbered clauses performs better at smaller chunk sizes because the retrievable unit is genuinely small. Research notes perform better at larger sizes because the argument spans paragraphs and a fragment loses the reasoning.

Every chunk carries a contextual header synthesised at index time, comprising the document title, the section path, and the effective date. This header is prepended to the chunk before embedding and is retained when the passage is passed to the model. The technique costs a small amount of index size and materially improves both retrieval quality and the model's ability to attribute what it read, and it is the single highest-return change the retrieval team made.

## Hybrid retrieval

Quarry retrieves using both dense and lexical signals and fuses the results. Dense retrieval handles paraphrase and conceptual similarity; lexical retrieval handles exact identifiers, product names, clause numbers, and the many internal acronyms that no general-purpose embedding model has ever seen. In a bank the lexical signal carries more weight than the published literature would suggest, because so many queries turn on an exact token.

Fusion is by reciprocal rank rather than by score normalisation. Score normalisation across heterogeneous retrievers proved unstable as corpora changed, producing quality regressions that were difficult to attribute. Reciprocal rank fusion is less theoretically satisfying and considerably more robust, which is the correct trade for a production system whose corpora shift weekly.

A reranking stage follows fusion for corpora where precision at low k matters. The reranker is a cross-encoder served through Lattice under the reranking capability class, and it is applied to the top fifty fused candidates to produce the final eight. Reranking is skipped for latency-sensitive applications and for corpora where evaluation shows it does not help, and the decision is recorded per corpus rather than applied globally.

## Refresh and staleness

Each corpus declares a refresh cadence and a staleness tolerance. Policy corpora refresh within one hour of source change because a superseded policy in the index is a live compliance exposure. Research corpora refresh nightly. Archival corpora refresh weekly. The declared tolerance is monitored, and a corpus that exceeds it raises an alert to the corpus owner rather than to the platform team, since the failure is almost always upstream.

Deletion propagates faster than insertion by design. A document withdrawn at source is removed from the index on the next propagation cycle regardless of the corpus refresh cadence, on a dedicated fast path. The asymmetry reflects the asymmetry of harm: a missing recent document produces an incomplete answer, while a present withdrawn document produces a wrong one.

Applications receive the index version alongside their results and are expected to surface effective dates to users. An answer assembled from passages effective at different dates is a known failure mode, and the mitigation is transparency rather than suppression, because the alternative of silently preferring recent passages produces answers that are confidently wrong about historical questions.

## Entitlements in practice

Entitlement resolution happens per request against the acting user's live entitlements, not against a cached snapshot. Caching entitlements was considered for latency reasons and rejected, because the window between a revocation and a cache expiry is exactly the window in which a departing employee retrieves what they should not. The latency cost is roughly eight milliseconds and is regarded as well spent.

Quarry never returns a partial passage to satisfy an entitlement constraint. A passage is either retrievable in full or not a candidate. Redacting within a passage was attempted early and abandoned, because the redacted output was frequently more misleading than an absence, and because the redaction boundaries themselves leaked structure.

Applications cannot widen their own entitlement scope. An application acting on behalf of a user retrieves what that user could retrieve and nothing more, and applications that require broader access must be registered as such and are then subject to H1 or H2 treatment regardless of what they do with the results. Several teams have discovered that their proposed architecture was tier-inflating and redesigned to act strictly on user authority instead.

## Failure behaviour

When retrieval returns nothing above the relevance floor, Quarry returns an empty result rather than the best of a poor set. This is a deliberate choice and one that applications must handle. Returning weak passages produces answers that appear grounded and are not, which is the most damaging failure mode a retrieval-augmented system has.

Applications are expected to detect the empty result and respond by declining to answer rather than by answering from parametric knowledge. Plumb includes a dedicated adversarial category for out-of-corpus questions precisely to test this behaviour, and a system that answers confidently when retrieval returned nothing fails its release gate regardless of how well it performs elsewhere.

Partial retrieval failure, where one corpus in a multi-corpus query is unavailable, is surfaced explicitly rather than absorbed. The application learns which corpora contributed, and an application that requires a corpus that is down should degrade visibly. Silent degradation across corpus availability was responsible for a class of incidents in which answers quietly became less complete for hours without any alert firing.
