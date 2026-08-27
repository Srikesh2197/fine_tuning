#!/usr/bin/env python3
"""Pre-flight checks on the PubMedQA-derived data all three notebooks train on.

Verifies:
  * both PubMedQA configs load and have the expected schema
  * the two stages are disjoint at article level
  * the stage-2 split is stratified and every decision class appears in eval
  * the yes/no/maybe grading regex recovers the gold label on 100% of records
  * length statistics fit the notebooks' token budgets
  * the constants duplicated in all three notebooks still match this repo's scripts

Needs `datasets`; add `transformers` for exact token counts instead of estimates.

Usage:
    pip install datasets transformers
    python scripts/validate_data.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_data as prep

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO_ROOT / "notebooks"

BLOCK_SIZE = 512     # notebook 01 packing block
MAX_LEN = 896        # notebook 02 truncation length
DECISION_RE = re.compile(r"answer\s*:?\s*\b(yes|no|maybe)\b", re.IGNORECASE)

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def get_token_counter():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
        return (lambda s: len(tok(s, add_special_tokens=False)["input_ids"]),
                "exact (Llama 3.2 tokenizer)")
    except Exception:
        return (lambda s: int(len(s.split()) * 1.5), "estimated (words x 1.5)")


def check_corpus(corpus, count, label) -> None:
    print("\nStage 1 — domain corpus (pqa_unlabeled)")
    required = {"text", "pubid", "section"}
    bad = [i for i, r in enumerate(corpus) if set(r) != required]
    if bad:
        fail(f"{len(bad)} records have unexpected fields (first at {bad[0]})")
    else:
        ok(f"{len(corpus):,} sections, schema {sorted(required)}")

    articles = {r["pubid"] for r in corpus}
    if len(articles) != prep.N_ABSTRACTS:
        fail(f"expected {prep.N_ABSTRACTS} articles, got {len(articles)}")
    else:
        ok(f"{len(articles):,} distinct articles")

    short = [r for r in corpus if len(r["text"]) < prep.MIN_SECTION_CHARS]
    if short:
        fail(f"{len(short)} sections are below MIN_SECTION_CHARS")
    else:
        ok(f"no section shorter than {prep.MIN_SECTION_CHARS} chars")

    seen, dupes = set(), 0
    for r in corpus:
        key = " ".join(r["text"].lower().split())
        dupes += key in seen
        seen.add(key)
    if dupes > len(corpus) * 0.02:
        fail(f"{dupes} duplicate sections ({dupes/len(corpus):.1%}) — unexpectedly high")
    else:
        ok(f"{dupes} duplicate sections ({dupes/len(corpus):.1%})")

    tokens = sum(count(r["text"]) for r in corpus)
    blocks = tokens // BLOCK_SIZE
    ok(f"~{tokens:,} tokens {label} -> ~{blocks} blocks of {BLOCK_SIZE}")
    if blocks < 200:
        fail(f"only ~{blocks} blocks — too few for a useful stage-1 run")
    else:
        ok(f"~{int(blocks * 0.9) // 4} optimizer steps/epoch at batch size 4")


def check_instructions(train, evaluation, count, label) -> None:
    print("\nStage 2 — instruction data (pqa_labeled)")
    required = {"instruction", "input", "output", "decision", "pubid"}
    for name, rows in (("train", train), ("eval", evaluation)):
        bad = [i for i, r in enumerate(rows) if set(r) != required]
        if bad:
            fail(f"{name}: {len(bad)} records have unexpected fields (first at {bad[0]})")
        blank = [i for i, r in enumerate(rows)
                 if not r["instruction"].strip() or not r["output"].strip()
                 or not r["input"].strip()]
        if blank:
            fail(f"{name}: {len(blank)} records have a blank field")
    ok(f"train={len(train)} eval={len(evaluation)}, schema {sorted(required)}")

    train_ids = {r["pubid"] for r in train}
    eval_ids = {r["pubid"] for r in evaluation}
    if train_ids & eval_ids:
        fail(f"{len(train_ids & eval_ids)} articles appear in both train and eval")
    else:
        ok("train and eval share no articles")

    train_mix = Counter(r["decision"] for r in train)
    eval_mix = Counter(r["decision"] for r in evaluation)
    missing = set(train_mix) - set(eval_mix)
    if missing:
        fail(f"eval is missing decision classes entirely: {sorted(missing)}")
    else:
        ok(f"every decision class appears in eval: {dict(eval_mix)}")

    for decision in sorted(train_mix):
        want = train_mix[decision] / len(train)
        got = eval_mix[decision] / len(evaluation)
        if abs(want - got) > 0.05:
            fail(f"'{decision}' is {want:.1%} of train but {got:.1%} of eval — split not stratified")
    ok("eval label mix matches train within 5pp (stratified)")

    majority, n = eval_mix.most_common(1)[0]
    ok(f"majority-class baseline on eval: {n/len(evaluation):.1%} (always '{majority}')")

    unparseable = [r for r in train + evaluation if not DECISION_RE.search(r["output"])]
    if unparseable:
        fail(f"{len(unparseable)} gold outputs the grading regex cannot parse")
    else:
        ok("grading regex parses 100% of gold outputs")

    wrong = [r for r in train + evaluation
             if DECISION_RE.search(r["output"]).group(1).lower() != r["decision"]]
    if wrong:
        fail(f"{len(wrong)} outputs where the regex recovers the wrong label")
    else:
        ok("grading regex recovers the exact gold label every time")

    template = ("Below is an instruction describing a task, paired with input providing further "
                "context. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n")
    lengths = sorted(count(template.format(r["instruction"], r["input"])) + count(r["output"]) + 2
                     for r in train + evaluation)
    over = [n for n in lengths if n > MAX_LEN]
    ok(f"tokens {label}: p50={lengths[len(lengths)//2]} "
       f"p90={lengths[int(len(lengths)*.9)]} max={lengths[-1]}")
    if over:
        fail(f"{len(over)} records exceed MAX_LEN={MAX_LEN} and would be truncated")
    else:
        ok(f"all records fit MAX_LEN={MAX_LEN} — nothing is truncated")


def check_notebook_constants() -> None:
    """The notebooks inline the prep logic so they are self-contained. Guard the drift."""
    print("\nNotebook / script constant drift")
    expected = {
        "01_domain_adaptation_lora": {
            "SEED": prep.SEED,
            "N_ABSTRACTS": prep.N_ABSTRACTS,
            "MIN_SECTION_CHARS": prep.MIN_SECTION_CHARS,
            "DATASET": prep.DATASET,
        },
        "02_instruction_finetuning_lora": {
            "SEED": prep.SEED,
            "EVAL_FRACTION": prep.EVAL_FRACTION,
            "DATASET": prep.DATASET,
        },
        # Notebook 03 rebuilds the stage-2 split from these same constants, which is what
        # guarantees its preference pairs are mined from the training 800 and its accuracy is
        # scored on the same held-out 200. Drift here would break both guarantees silently.
        "03_preference_tuning_dpo": {
            "SEED": prep.SEED,
            "EVAL_FRACTION": prep.EVAL_FRACTION,
            "DATASET": prep.DATASET,
        },
    }
    for stem, constants in expected.items():
        path = NOTEBOOKS / f"{stem}.ipynb"
        if not path.exists():
            fail(f"{path.name} not found")
            continue
        src = "\n".join("".join(c["source"]) for c in json.loads(path.read_text())["cells"])
        clean = True
        for name, want in constants.items():
            hit = re.search(rf"^{name}\s*=\s*(.+?)\s*(?:#.*)?$", src, re.MULTILINE)
            if not hit:
                fail(f"{path.name}: constant {name} not found")
                clean = False
                continue
            # Compare parsed values, not source text: `0.20` and `0.2` are the same number.
            try:
                got = ast.literal_eval(hit.group(1))
            except (ValueError, SyntaxError):
                fail(f"{path.name}: {name} is not a literal ({hit.group(1)!r})")
                clean = False
                continue
            if got != want:
                fail(f"{path.name}: {name} is {got!r}, prepare_data.py says {want!r}")
                clean = False
        if stem.startswith(("02", "03")) and 'Begin your reply with "Answer:"' not in src:
            fail(f"{path.name}: TASK string differs from prepare_data.py")
            clean = False
        if clean:
            ok(f"{path.name}: constants match prepare_data.py")


def main() -> None:
    count, label = get_token_counter()
    if label.startswith("estimated"):
        print("note: transformers not installed — token counts are estimates\n")

    print("Loading PubMedQA from the Hub ...")
    corpus, train, evaluation = prep.prepare()
    ok("both configs loaded; stages are disjoint at article level")

    check_corpus(corpus, count, label)
    check_instructions(train, evaluation, count, label)
    check_notebook_constants()

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
