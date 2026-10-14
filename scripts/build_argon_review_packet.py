#!/usr/bin/env python3
"""Bundle the current MLRescurix v2 planning docs for external review."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "v2"
OUTPUT = DOCS / "ARGON_REVIEW_PACKET.txt"

PROMPT = """ARGON REVIEW REQUEST — MLRESCURIX V2 DOCUMENTS

Please review the repository audit and phased implementation plan below against
the two original Argon blueprints and the repository facts quoted in the packet.
The packet also includes your returned planning review and our REVIEW_RESPONSE.md
disposition. Check whether the merged changes resolve those findings and whether
our corrections are justified; do not simply repeat the previous replacement files.
Your job is to suggest improvements or provide replacement document text that I
can use to overwrite the corresponding planning documents in my repository.

Review for:
1. Incorrect or stale codebase claims and missed relevant subsystems.
2. Unsafe or scientifically invalid boundaries, especially labels, splits,
   evaluation, replay/recovery, artifacts, resource limits and finalization.
3. Missing dependencies or acceptance tests between phases.
4. A practical, incremental implementation sequence for an agent-first ML coding
   harness with transparent artifacts, bounded local compute, durable recovery,
   and a deployable verified Apex model.
5. Useful additions for model rescue, tabular ML first, and later capability-based
   support for other ML tasks, accelerators or remote workers.
6. Claims or performance numbers that need to be removed, narrowed or measured.

Please do not treat either original blueprint as authoritative: check its claims
against the audit in this packet. Do not silently weaken existing security,
scientific, release or recovery invariants. Distinguish verified repository facts,
recommendations, assumptions and open questions. Avoid promising universal ML,
zero-token recovery, zero-copy memory, exact resource prediction, or fixed speedups
without evidence. Point out internal contradictions and recommend a clear resolution.

RESPONSE FORMAT

First give a concise review: strongest parts, corrections, critical gaps and any
questions that truly block a sound plan. Then return replacement contents for each
document you recommend changing, using this exact format:

<<<FILE: docs/v2/README.md>>>
complete replacement file text
<<<END FILE>>>

Use a separate FILE block for every replacement. Return complete files, not diffs
or excerpts, so I can overwrite them directly. Include every file whose content
you want changed, and no files whose content should stay as-is. Preserve stable
phase/task IDs where possible. Add a short change log explaining important changes
and why. If you recommend no edits, say so and return no FILE blocks. Do not claim
you tested source code unless you actually did; this packet contains documents, not
a live checkout for you to modify.

The packet below contains all current docs/v2 Markdown files, the original
blueprints and the returned planning review. It is a static review packet,
not a grant of access to mutate the repo.
"""


ORDER = [
    "README.md",
    "PROGRESS.md",
    "AUDIT.md",
    "ROADMAP.md",
    "ARCHITECTURE.md",
    "DECISIONS.md",
    "SESSION_PROMPT.md",
    "REVIEW_RESPONSE.md",
    "ARGON_BLUEPRINT.md",
    "ARGON_REPO_BLUEPRINT.md",
    "ARGON_PLAN_REVIEW.txt",
]


def main() -> None:
    chunks = [PROMPT.rstrip(), "\n\n" + "=" * 78 + "\nDOCUMENTS FOR REVIEW\n" + "=" * 78]
    for name in ORDER:
        path = DOCS / name
        if not path.is_file():
            raise SystemExit(f"Required source document is missing: {path}")
        content = path.read_text(encoding="utf-8").rstrip()
        chunks.append(f"\n\n<<<SOURCE FILE: docs/v2/{name}>>>\n\n{content}\n\n<<<END SOURCE FILE>>>")
    OUTPUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes, {len(ORDER)} source files).")


if __name__ == "__main__":
    main()
