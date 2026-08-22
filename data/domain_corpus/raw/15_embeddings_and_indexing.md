# Embeddings, Indexing, and Domain Vocabulary

## Embedding model selection

Meridian Trust selects embedding models on retrieval performance against the firm's own labelled query sets rather than on published benchmark results. Public benchmarks are useful for shortlisting and misleading for selection, because the firm's corpora are dense with internal terminology, product names, and clause references that no public benchmark contains.

The evaluation compares candidate embedding models on recall at k and distinct-source recall across every registered corpus, because performance varies by corpus more than teams expect. An embedding model that leads on policy text may trail on research notes, and the platform supports per-corpus embedding model selection precisely because a single global choice is usually a compromise.

Dimensionality is chosen against measured retrieval quality and index cost jointly. Higher dimensionality improves retrieval marginally and increases index size, memory footprint, and query latency linearly. The firm's experience is that the quality curve flattens well before the cost curve does, and several corpora run at reduced dimensionality with no measurable retrieval loss.

Changing an embedding model requires reindexing the entire corpus and produces a new Ledgerline identifier. Mixed-embedding indices are prohibited, because similarity scores from different embedding spaces are not comparable and fusing them produces rankings that are arbitrary in a way that is difficult to detect.

## Domain vocabulary

Internal terminology is the dominant retrieval challenge in a bank. A query mentioning a Halton tier, a Blue-file, or a Gatepost decision contains tokens that a general-purpose embedding model has never seen in this sense, and dense retrieval on such queries degrades toward matching general semantic similarity rather than the specific concept.

The firm addresses this in three ways. Hybrid retrieval carries the lexical signal, which handles exact terminology reliably. Contextual chunk headers place the document title and section path into the embedded text, which supplies surrounding context that partially disambiguates unfamiliar terms. And for the highest-volume corpora, the embedding model itself is fine-tuned on the firm's terminology.

Embedding fine-tuning uses contrastive training on query and passage pairs derived from the Lattice audit log, with positives drawn from passages that produced satisfactory answers and negatives sampled from high-scoring passages that did not. The resulting improvement is concentrated exactly where it is needed, on queries containing internal terminology, and is negligible on general queries.

Embedding adapters are governed like any other adapter. They are registered in Sable as components of the retrieval configuration, they are pinned to a base embedding model version, and changing one invalidates every index built with it. The reindexing cost is why embedding fine-tuning is reserved for corpora where the retrieval quality gain justifies it.

## Index structure and refresh

Quarry uses an approximate nearest neighbour index tuned per corpus for the recall and latency the corpus requires. Interactive corpora accept a small recall loss for latency; batch corpora do not. The parameters are recorded in the corpus configuration and form part of the Ledgerline identifier, because a corpus indexed at different parameters retrieves differently and evaluations across the two are not comparable.

Incremental indexing handles document insertion and update; deletion propagates through a separate fast path as described in the Quarry standard. Full reindexing occurs on embedding model change, on chunking configuration change, and on a scheduled cadence that varies by corpus, to reclaim the quality that incremental updates gradually erode.

Index refresh is monitored against the corpus staleness tolerance. The monitored quantity is the age of the oldest unpropagated source change rather than the time since the last refresh run, because a refresh that ran and processed nothing is not evidence of freshness.

## Chunk size and its effects

Chunk size interacts with everything downstream and is the parameter teams tune last and should tune first. Smaller chunks retrieve more precisely and supply less context; larger chunks supply more context and dilute the embedding, so that a passage about one thing embedded together with three other things matches queries about none of them well.

The firm's default of four hundred tokens with sixty tokens of overlap is a starting point, not a recommendation. Policy corpora with numbered clauses perform better at half that size. Research corpora perform better at double. The correct approach is to evaluate three or four configurations against the corpus's labelled query set, which costs a day and is skipped far more often than it should be.

Overlap exists to prevent a relevant span being split across a boundary and consequently matching neither chunk well. It costs index size proportional to the overlap fraction and it introduces near-duplicate results, which is why distinct-source recall matters as a metric and why deduplication runs on the fused candidate set before reranking.

## Reranking

Reranking applies a cross-encoder to the fused candidate set, scoring each candidate jointly with the query rather than independently. It is markedly more accurate than bi-encoder retrieval and markedly more expensive, which is why it is applied to a shortlist rather than to the corpus.

Meridian applies reranking to the top fifty fused candidates to produce the final eight passages. The shortlist size matters: too small and reranking cannot recover a relevant passage the first stage ranked poorly, too large and the latency cost dominates. Fifty was selected empirically and is overridden per corpus where evaluation supports it.

Reranking is skipped where evaluation shows no benefit, which happens more often than expected on corpora with strong lexical signal and homogeneous document structure. The decision is recorded per corpus with the supporting evaluation, rather than applied as a global default in either direction.

## Evaluating index changes

An index change is evaluated with the same rigour as a model change and passes through the same Plumb gate. This surprises teams who regard indexing as infrastructure, and it follows directly from the position that the corpus and the retrieval configuration are part of the registered model rather than adjacent to it.

The evaluation runs retrieval metrics against the labelled query set and end-to-end metrics against the full suite, because retrieval improvements do not always survive to the answer. A configuration that improves recall at k while returning longer passages can degrade end-to-end quality by crowding the context window, and only end-to-end evaluation catches it.

Index changes are also the most common cause of unexplained production quality movement, because they are frequently made by a different team on a different schedule from the application team that experiences the effect. Quarry therefore emits an index version change event that applications subscribe to, and continuous evaluation is triggered automatically on receipt.
