# Evaluating Retrieval-Augmented Systems

## Separating the stages

A retrieval-augmented system fails in two distinct ways, and evaluating it as a single black box makes those failures indistinguishable. Meridian Trust therefore evaluates retrieval and generation separately before evaluating them jointly, and every Plumb suite for a retrieval-augmented application reports all three layers.

The separation matters operationally. An end-to-end score that falls tells the team something is wrong. A retrieval score that falls while generation quality holds tells the team the index refresh broke. The diagnostic value of layered measurement is worth the additional effort of maintaining separate labelled sets, and teams that skip it spend the saved effort several times over during incidents.

## Retrieval metrics

Retrieval is evaluated with recall at k, mean reciprocal rank, and normalised discounted cumulative gain, computed against a labelled set of queries with known relevant passages. Recall at k is the primary metric for retrieval-augmented generation because the generator cannot use what it never received, and the value of k is set to the number of passages actually passed to the model rather than to a conventional ten.

Recall at k understates a specific and common failure: the relevant passage is retrieved but ranked below several near-duplicates, crowding the context window. Meridian therefore also reports distinct-source recall, which counts how many distinct source documents among the required set appear in the returned passages. A retrieval configuration that returns eight chunks from the same document scores well on plain recall and poorly on the metric that matters.

Labelled sets are built by sampling real queries from the Lattice audit log, having subject-matter experts identify the passages that should have been retrieved, and freezing the result with a Ledgerline version. They are expensive, and the firm's experience is that a set of two hundred well-labelled queries is worth more than two thousand cheaply labelled ones, because noisy labels produce metric movements that teams then chase.

## Grounding and faithfulness

Generation quality for retrieval-augmented systems is assessed principally on grounding: whether every factual assertion in the output is supported by a retrieved passage. Grounding is measured by decomposing the output into atomic claims and checking each against the provided context, a process performed by an evaluation model under the LLM-as-judge controls described in the Plumb standard.

Grounding is scored per claim rather than per response, and the headline metric is the proportion of claims that are supported. Per-response scoring conceals the difference between a response with one unsupported claim among twelve and a response that is entirely fabricated, and those two failures warrant very different responses from the team.

Unsupported claims are further classified as contradicted, unsupported but plausible, or unsupported and implausible. Contradicted claims, where the retrieved passage says the opposite, are treated as severe and block release at any tier. The other categories are tolerated at declining thresholds by Halton tier, with H1 systems held to the tightest standard.

## Answerability and abstention

A substantial fraction of production queries have no answer in the corpus, and a system that always answers is worse than one that sometimes declines. Meridian evaluates abstention explicitly, using a labelled set in which a known proportion of queries are unanswerable from the corpus, and reports both the abstention rate on unanswerable queries and the false abstention rate on answerable ones.

The two rates trade against each other, and the correct operating point differs by application. A system supporting a regulated advice process should abstain aggressively; a system supporting internal document search should not, because a cautious search tool is a useless one. Blue-files are required to state the chosen operating point and the reasoning, rather than reporting whichever rate flatters the system.

Abstention behaviour degrades in a characteristic way when a system is fine-tuned on instruction data that contains no abstention examples. The model learns that every question has an answer, because in its training data every question did. Instruction sets for retrieval-augmented systems at Meridian are therefore required to include abstention examples, typically at around one in ten records.

## Attribution

Where a system surfaces citations, the citations are evaluated independently of the claims. A response can be fully grounded and badly cited, and users trust citations more than they trust prose, which makes a wrong citation more damaging than a wrong sentence. Attribution accuracy is measured as the proportion of citations that actually support the claim they are attached to.

The common failure is citation drift, where a response cites the correct document but the wrong passage within it, or cites the passage that motivated an earlier sentence rather than the one supporting the sentence it is attached to. This is invisible to users who do not follow the citation and corrosive to those who do, and it is caught only by explicit per-citation evaluation.

## End-to-end evaluation

End-to-end evaluation uses held-out real queries scored by subject-matter experts against a rubric covering correctness, completeness, grounding, and usefulness. It is expensive, slow, and irreplaceable. Automated proxies correlate well enough to gate routine releases and poorly enough that a system optimised solely against them drifts toward outputs that score well and help nobody.

Meridian's practice is to run automated evaluation on every change and expert evaluation at every revalidation and before every H1 or H2 release. The expert set is also used to recalibrate the automated judges, and a divergence between automated and expert scores is investigated as a defect in the evaluation apparatus rather than dismissed.

## What changes when the base model is fine-tuned

Fine-tuning a base model that sits inside a retrieval-augmented system introduces a specific evaluation obligation: demonstrating that the model still prefers the retrieved context over its own parameters. Domain fine-tuning makes the model more confident about the domain, and a model confident about the domain is more willing to answer from memory when retrieval is weak.

This is measured with a deliberately constructed counterfactual set, in which retrieved passages state something that contradicts what the base model would otherwise say. A well-behaved system follows the passage. A system that has absorbed the domain too enthusiastically follows its parameters, and the counterfactual set is the only reliable way to detect the difference before production does.

The counterfactual set is also the mechanism by which the firm detects that a fine-tune has memorised training passages verbatim. A model reproducing corpus text that was not retrieved has leaked training data into the response path, which for a corpus containing client material is a confidentiality incident rather than a quality finding, and is escalated accordingly.
