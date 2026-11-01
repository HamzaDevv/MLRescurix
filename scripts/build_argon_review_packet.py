#!/usr/bin/env python3
"""Bundle the current MLRescurix master planning, audit, review, and handoff docs for external review."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
PLANS_DIR = ROOT / "Major_refactor_docs" / "plans"
OUTPUT = DOCS_DIR / "ARGON_REVIEW_PACKET.txt"

PROMPT = """AUTONOMOUS MODEL CREATOR REVIEW REQUEST

Please review the master refactor plan and companion architectural documents
against the implementation progress and repository ground truth.
Check whether the implementation resolves findings from the independent review,
enforces strict boundaries on data, evaluation, provider routing, and resource
accounting, and preserves transparent reproducible model bundles.
"""

REVIEW_DOCUMENTS = [
    PLANS_DIR / "FINAL_AUTONOMOUS_MODEL_CREATOR_MASTER_PLAN.md",
    PLANS_DIR / "AGENTIC_WORKFLOW_AND_FLEXIBILITY_AUDIT.md",
    DOCS_DIR / "CORE_ARCHITECTURE_GROUND_TRUTH.md",
    DOCS_DIR / "ARCHITECTURE.md",
    DOCS_DIR / "DECISIONS.md",
    DOCS_DIR / "AUTONOMOUS_MODEL_CREATOR_SESSION_1_HANDOFF.md",
    DOCS_DIR / "AUTONOMOUS_MODEL_CREATOR_SESSION_1_REVIEW.md",
    DOCS_DIR / "AUTONOMOUS_MODEL_CREATOR_SESSION_2_HANDOFF.md",
]


def main() -> None:
    chunks = [PROMPT.rstrip(), "\n\n" + "=" * 78 + "\nDOCUMENTS FOR REVIEW\n" + "=" * 78]
    found_files = []
    for path in REVIEW_DOCUMENTS:
        if not path.is_file():
            continue
        found_files.append(path)
        content = path.read_text(encoding="utf-8").rstrip()
        rel_path = path.relative_to(ROOT)
        chunks.append(f"\n\n<<<SOURCE FILE: {rel_path}>>>\n\n{content}\n\n<<<END SOURCE FILE>>>")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes, {len(found_files)} source files bundled of {len(REVIEW_DOCUMENTS)} configured).")


if __name__ == "__main__":
    main()
