#!/usr/bin/env python3
"""Prepare both fine-tuning stages from a single real dataset: PubMedQA.

    https://huggingface.co/datasets/qiaojin/PubMedQA   (MIT licence)

One source, two stages:

  stage 1 (notebook 01)  pqa_unlabeled -> raw biomedical abstract prose
                         Non-instructional fine-tuning: plain next-token
                         prediction over structured PubMed abstracts.

  stage 2 (notebook 02)  pqa_labeled   -> (instruction, input, output) triples
                         Expert-annotated question / abstract / yes-no-maybe
                         with a written justification.

The two configs are disjoint by construction (different PubMed articles), and
this script asserts that rather than trusting it.

The notebooks do NOT read the files this script writes — they call
load_dataset() and run this same logic inline, so they are self-contained and
need no repo checkout. This script exists so you can materialise and inspect
the data locally without a GPU.

IMPORTANT: the constants below are duplicated in both notebooks. validate_data.py
checks that they still agree.

Usage:
    pip install datasets
    python scripts/prepare_data.py
    python scripts/prepare_data.py --out data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --- constants mirrored in both notebooks -----------------------------------
DATASET = "qiaojin/PubMedQA"
SEED = 20260815
N_ABSTRACTS = 1000        # stage-1 abstracts sampled from pqa_unlabeled
MIN_SECTION_CHARS = 120   # drop abstract sections shorter than this
EVAL_FRACTION = 0.20      # stage-2 held-out share -> 800 train / 200 eval

TASK = (
    "Answer the research question using only the abstract provided. "
    'Begin your reply with "Answer:" followed by yes, no, or maybe, '
    "then justify it in one or two sentences."
)
# ----------------------------------------------------------------------------


def join_sections(context: dict) -> str:
    """Render an abstract's sections as `LABEL: text` blocks."""
    labels = context.get("labels") or []
    contexts = context.get("contexts") or []
    parts = []
    for i, text in enumerate(contexts):
        text = " ".join(text.split())
        if not text:
            continue
        label = (labels[i] if i < len(labels) else "").strip()
        parts.append(f"{label}: {text}" if label else text)
    return "\n\n".join(parts)


def build_corpus(unlabeled) -> list[dict]:
    """Stage 1: one record per abstract section of raw prose."""
    sampled = unlabeled.shuffle(seed=SEED).select(range(N_ABSTRACTS))
    records = []
    for row in sampled:
        labels = row["context"].get("labels") or []
        for i, text in enumerate(row["context"]["contexts"]):
            text = " ".join(text.split())
            if len(text) < MIN_SECTION_CHARS:
                continue
            records.append({
                "text": text,
                "pubid": row["pubid"],
                "section": (labels[i] if i < len(labels) else "").strip() or "UNLABELLED",
            })
    return records


def build_instructions(labeled) -> list[dict]:
    """Stage 2: Alpaca-style triples with a gradable yes/no/maybe decision."""
    records = []
    for row in labeled:
        abstract = join_sections(row["context"])
        if not abstract:
            continue
        long_answer = " ".join(row["long_answer"].split())
        records.append({
            "instruction": f'{TASK}\n\nQuestion: {row["question"].strip()}',
            "input": abstract,
            "output": f'Answer: {row["final_decision"]}\n\n{long_answer}',
            "decision": row["final_decision"],
            "pubid": row["pubid"],
        })
    return records


def split_stratified(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hold out EVAL_FRACTION, stratified by decision so eval keeps the label mix.

    `maybe` is only 11% of the data; an unstratified split would leave the eval
    set with an unstable number of them and make accuracy noisy.
    """
    rng = random.Random(SEED + 1)
    by_decision: dict[str, list[dict]] = {}
    for r in records:
        by_decision.setdefault(r["decision"], []).append(r)

    train, evaluation = [], []
    for decision in sorted(by_decision):
        rows = sorted(by_decision[decision], key=lambda r: r["pubid"])
        rng.shuffle(rows)
        n_eval = round(len(rows) * EVAL_FRACTION)
        evaluation.extend(rows[:n_eval])
        train.extend(rows[n_eval:])

    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation


def prepare():
    from datasets import load_dataset

    print(f"Loading {DATASET} ...")
    unlabeled = load_dataset(DATASET, "pqa_unlabeled", split="train")
    labeled = load_dataset(DATASET, "pqa_labeled", split="train")
    print(f"  pqa_unlabeled {len(unlabeled):,} rows")
    print(f"  pqa_labeled   {len(labeled):,} rows")

    corpus = build_corpus(unlabeled)
    instructions = build_instructions(labeled)
    train, evaluation = split_stratified(instructions)

    # The two stages must not share articles, or stage 2's eval measures what
    # stage 1 already memorised.
    corpus_ids = {r["pubid"] for r in corpus}
    instruction_ids = {r["pubid"] for r in instructions}
    overlap = corpus_ids & instruction_ids
    assert not overlap, f"{len(overlap)} articles appear in BOTH stages: {sorted(overlap)[:5]}"

    return corpus, train, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    args = parser.parse_args()

    corpus, train, evaluation = prepare()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("corpus.jsonl", corpus),
                       ("instruct_train.jsonl", train),
                       ("instruct_eval.jsonl", evaluation)]:
        (out / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )

    from collections import Counter
    words = sum(len(r["text"].split()) for r in corpus)
    print(f"\nWrote {out}/")
    print(f"  corpus.jsonl          {len(corpus):,} sections from "
          f"{len({r['pubid'] for r in corpus}):,} abstracts")
    print(f"                        {words:,} words, ~{int(words * 1.35):,} tokens")
    print(f"  instruct_train.jsonl  {len(train):,}  {dict(Counter(r['decision'] for r in train))}")
    print(f"  instruct_eval.jsonl   {len(evaluation):,}  {dict(Counter(r['decision'] for r in evaluation))}")
    majority = Counter(r["decision"] for r in evaluation).most_common(1)[0]
    print(f"\n  majority-class baseline on eval: {majority[1] / len(evaluation):.1%} "
          f"(always answer '{majority[0]}')")


if __name__ == "__main__":
    main()
