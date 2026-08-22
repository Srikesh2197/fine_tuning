#!/usr/bin/env python3
"""Validate the committed datasets before either notebook consumes them.

Checks performed:
  * corpus.jsonl and the instruction files parse as JSONL with the right schema
  * committed artifacts match a fresh build of the generator scripts
  * no duplicate prompts within a split
  * no prompt or free-text answer leaks from train into eval
  * length statistics stay inside the 512-token budget the notebooks assume

Runs on the standard library alone. If `transformers` happens to be installed
it uses the real Llama 3.2 tokenizer for exact counts; otherwise it falls back
to a words x 1.4 estimate and says so.

Usage:
    python scripts/validate_data.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_instructions import normalise  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
CORPUS = REPO_ROOT / "data" / "domain_corpus" / "corpus.jsonl"
TRAIN = REPO_ROOT / "data" / "instruction" / "train.jsonl"
EVAL = REPO_ROOT / "data" / "instruction" / "eval.jsonl"

BLOCK_SIZE = 512          # stage-1 packing block size
MAX_INSTRUCTION_LEN = 512  # stage-2 truncation length

failures: list[str] = []
notes: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print(f"  FAIL  {message}")


def ok(message: str) -> None:
    print(f"  ok    {message}")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        fail(f"{path.relative_to(REPO_ROOT)} does not exist")
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            fail(f"{path.name}:{n} is not valid JSON ({exc})")
    return rows


def get_token_counter():
    """Return (fn, label). Exact tokenizer if available, estimate otherwise."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
        return (lambda s: len(tok(s, add_special_tokens=False)["input_ids"]),
                "exact (Llama 3.2 tokenizer)")
    except Exception:
        return (lambda s: int(len(s.split()) * 1.4), "estimated (words x 1.4)")


def check_freshness() -> None:
    print("\nFreshness (committed artifacts vs a fresh build)")
    for script in ("build_corpus.py", "generate_instructions.py"):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / script), "--check"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            ok(f"{script} — committed output is current")
        else:
            fail(f"{script} — {(result.stderr or result.stdout).strip()}")


def check_corpus(count_tokens, label: str) -> None:
    print("\nDomain corpus")
    rows = read_jsonl(CORPUS)
    if not rows:
        return

    required = {"text", "doc", "topic", "section"}
    bad = [i for i, r in enumerate(rows) if set(r) != required]
    if bad:
        fail(f"{len(bad)} corpus records have unexpected fields (first at index {bad[0]})")
    else:
        ok(f"{len(rows)} records, schema {sorted(required)}")

    empty = [i for i, r in enumerate(rows) if not r["text"].strip()]
    if empty:
        fail(f"{len(empty)} corpus records have empty text")

    seen: dict[str, int] = {}
    dupes = 0
    for r in rows:
        key = normalise(r["text"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            dupes += 1
    if dupes:
        fail(f"{dupes} duplicate paragraphs in the corpus")
    else:
        ok("no duplicate paragraphs")

    lengths = sorted(count_tokens(r["text"]) for r in rows)
    total = sum(lengths)
    p50 = lengths[len(lengths) // 2]
    p95 = lengths[int(len(lengths) * 0.95)]
    ok(f"tokens {label}: total ~{total:,}, per-paragraph p50={p50} p95={p95} max={lengths[-1]}")

    blocks = total // BLOCK_SIZE
    if blocks < 30:
        fail(f"only ~{blocks} blocks of {BLOCK_SIZE} tokens — too few for a useful run")
    else:
        ok(f"~{blocks} training blocks of {BLOCK_SIZE} tokens")

    docs = {r["doc"] for r in rows}
    ok(f"{len(docs)} source documents")


def check_instructions(count_tokens, label: str) -> None:
    print("\nInstruction data")
    train = read_jsonl(TRAIN)
    evaluation = read_jsonl(EVAL)
    if not train or not evaluation:
        return

    required = {"instruction", "input", "output", "kind", "topic"}
    for name, rows in (("train", train), ("eval", evaluation)):
        bad = [i for i, r in enumerate(rows) if set(r) != required]
        if bad:
            fail(f"{name}: {len(bad)} records have unexpected fields (first at {bad[0]})")
        blank = [i for i, r in enumerate(rows)
                 if not r["instruction"].strip() or not r["output"].strip()]
        if blank:
            fail(f"{name}: {len(blank)} records have a blank instruction or output")
    if not failures:
        ok(f"train={len(train)} eval={len(evaluation)}, schema {sorted(required)}")

    for name, rows in (("train", train), ("eval", evaluation)):
        keys = [(normalise(r["instruction"]), normalise(r["input"])) for r in rows]
        if len(keys) != len(set(keys)):
            fail(f"{name}: {len(keys) - len(set(keys))} duplicate prompts")
        else:
            ok(f"{name}: no duplicate prompts")

    train_keys = {(normalise(r["instruction"]), normalise(r["input"])) for r in train}
    overlap = [r for r in evaluation
               if (normalise(r["instruction"]), normalise(r["input"])) in train_keys]
    if overlap:
        fail(f"{len(overlap)} eval prompts also appear in train")
    else:
        ok("no prompt overlap between train and eval")

    free_text = {"qa", "excerpt", "abstain"}
    train_outputs = {normalise(r["output"]) for r in train if r["kind"] in free_text}
    answer_overlap = [r for r in evaluation
                      if r["kind"] in free_text and normalise(r["output"]) in train_outputs]
    if answer_overlap:
        fail(f"{len(answer_overlap)} eval answers also appear in train (free-text kinds)")
    else:
        ok("no free-text answer overlap between train and eval")

    kinds = sorted({r["kind"] for r in train})
    missing = [k for k in kinds if not any(r["kind"] == k for r in evaluation)]
    if missing:
        fail(f"eval set is missing these task kinds entirely: {missing}")
    else:
        ok(f"every task kind appears in eval: {kinds}")

    with_input = sum(1 for r in train + evaluation if r["input"].strip())
    ok(f"{with_input} records carry a non-empty `input` field")

    abstain = sum(1 for r in train if r["kind"] == "abstain")
    share = abstain / len(train)
    if share < 0.02:
        fail(f"only {share:.1%} of train is abstention data — too little to teach declining")
    else:
        ok(f"abstention examples are {share:.1%} of train ({abstain} records)")

    rendered = [
        count_tokens(f"{r['instruction']}\n{r['input']}\n{r['output']}")
        for r in train + evaluation
    ]
    rendered.sort()
    over = [n for n in rendered if n > MAX_INSTRUCTION_LEN - 40]
    p95 = rendered[int(len(rendered) * 0.95)]
    ok(f"tokens {label}: p50={rendered[len(rendered) // 2]} p95={p95} max={rendered[-1]}")
    if over:
        fail(f"{len(over)} records approach or exceed the {MAX_INSTRUCTION_LEN}-token budget")
    else:
        ok(f"all records fit the {MAX_INSTRUCTION_LEN}-token budget with headroom")


def main() -> None:
    count_tokens, label = get_token_counter()
    if label.startswith("estimated"):
        notes.append(
            "transformers is not installed locally, so token counts are estimates. "
            "That is fine for a pre-flight check; the notebooks count exactly."
        )

    check_freshness()
    check_corpus(count_tokens, label)
    check_instructions(count_tokens, label)

    print()
    for note in notes:
        print(f"note: {note}")

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
