#!/usr/bin/env python3
"""Build the domain-adaptation corpus from raw Markdown documents.

This mirrors the PDF -> text -> paragraphs pipeline that most "train on your own
documents" tutorials use, but starting from Markdown so the whole thing is
reproducible and diffable in git.

    raw/*.md  ->  paragraphs  ->  clean/filter  ->  corpus.jsonl

Each output record is one paragraph:

    {"text": ..., "doc": ..., "topic": ..., "section": ...}

`text` is the only field the training notebook uses. The others exist so you can
slice the corpus during EDA and so `generate_instructions.py` can build
input-bearing tasks with a deterministic answer.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --check   # verify committed file is current
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "domain_corpus" / "raw"
OUT_PATH = REPO_ROOT / "data" / "domain_corpus" / "corpus.jsonl"

# Paragraphs shorter than this are almost always stray fragments, not prose.
MIN_CHARS = 120

# Map a document stem to the internal system or standard it describes. Used by
# generate_instructions.py to build classification tasks with a known answer.
DOC_SUBJECT = {
    "01_model_risk_framework": "the Halton scale",
    "02_gatepost_change_control": "Gatepost",
    "03_blue_file_documentation": "the Blue-file",
    "04_lattice_llm_platform": "Lattice",
    "05_quarry_retrieval_service": "Quarry",
    "06_retrieval_evaluation": "retrieval-augmented evaluation",
    "07_plumb_eval_harness": "Plumb",
    "08_scrim_pii_controls": "Scrim",
    "09_finetuning_practice": "fine-tuning practice",
    "10_monitoring_drift": "production monitoring",
    "11_ledgerline_lineage": "Ledgerline",
    "12_validation_and_challenger": "independent validation",
    "13_llm_application_patterns": "application patterns",
    "14_prompt_standard": "the prompt engineering standard",
    "15_embeddings_and_indexing": "embeddings and indexing",
    "16_agentic_workflows": "tool use and agentic workflows",
    "17_vendor_and_open_weights": "third-party model risk",
    "18_incident_reviews": "incident reviews",
}


def topic_from_stem(stem: str) -> str:
    """`01_model_risk_framework` -> `model-risk-framework`."""
    return re.sub(r"^\d+_", "", stem).replace("_", "-")


def parse_document(path: Path) -> list[dict]:
    """Split one Markdown document into paragraph records.

    Headings are dropped from the text but retained as the `section` label of
    the paragraphs that follow them, so we keep the structural signal without
    training the model to emit `##`.
    """
    stem = path.stem
    topic = topic_from_stem(stem)
    records: list[dict] = []
    section = ""

    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        block = block.strip()
        if not block:
            continue

        if block.startswith("#"):
            # Track the most recent section heading; skip the document title.
            if block.startswith("## "):
                section = block[3:].strip()
            continue

        # Collapse soft-wrapped lines and runs of whitespace into single spaces.
        text = re.sub(r"\s+", " ", block).strip()
        if len(text) < MIN_CHARS:
            continue

        records.append(
            {"text": text, "doc": stem, "topic": topic, "section": section}
        )

    return records


def build() -> list[dict]:
    paths = sorted(RAW_DIR.glob("*.md"))
    if not paths:
        sys.exit(f"No Markdown documents found under {RAW_DIR}")

    records: list[dict] = []
    for path in paths:
        doc_records = parse_document(path)
        if not doc_records:
            sys.exit(f"{path.name} produced no paragraphs — check its formatting")
        records.extend(doc_records)

    unknown = {r["doc"] for r in records} - set(DOC_SUBJECT)
    if unknown:
        sys.exit(f"DOC_SUBJECT is missing entries for: {sorted(unknown)}")

    return records


def serialise(records: list[dict]) -> str:
    return "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in records
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed corpus.jsonl differs from a fresh build",
    )
    args = parser.parse_args()

    records = build()
    payload = serialise(records)

    if args.check:
        if not OUT_PATH.exists():
            sys.exit(f"{OUT_PATH} does not exist")
        if OUT_PATH.read_text(encoding="utf-8") != payload:
            sys.exit(f"{OUT_PATH} is stale — re-run `python scripts/build_corpus.py`")
        print(f"OK: {OUT_PATH.relative_to(REPO_ROOT)} matches a fresh build")
        return

    OUT_PATH.write_text(payload, encoding="utf-8")

    words = sum(len(r["text"].split()) for r in records)
    docs = len({r["doc"] for r in records})
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  documents  : {docs}")
    print(f"  paragraphs : {len(records)}")
    print(f"  words      : {words:,}")
    print(f"  est. tokens: ~{int(words * 1.35):,}")


if __name__ == "__main__":
    main()
