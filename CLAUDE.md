# Credentialing RAG

Cited Q&A over public healthcare provider-credentialing policy documents. Phased toward a
fuller architecture (structure-aware chunking, evaluation, router, metadata, stress tests);
see the README roadmap.

## Hard rules
- Answers come only from retrieved sources. An answer reaches the user only if it passes
  `check_answer` (every [n] is a retrieved source, or exactly NOT_FOUND). Prompts are not a
  safety mechanism on their own.
- Education only: no legal or compliance determinations about a specific provider.
- Source documents are supplied by the user. Never download or fetch documents. NCQA
  standards are copyrighted: never commit them.
- `domain/` and `application/` import no adapter and no SDK, so the whole pipeline is
  testable with fakes at $0.

## Conventions
- Python 3.12 + uv. Tests: `uv run pytest` (offline, free). Do not run paid model calls in
  tests; the user runs live checks themselves.
- Secrets in `.env` only (ANTHROPIC_API_KEY), never committed.
- Default answer model: Claude Haiku 5.5 (`claude-haiku-5-5`), overridable via RAG_MODEL.
- Every wrong answer found in use becomes a line in `evals/golden.jsonl`.
- Tag versions v0.0, v0.1... at each ship check.
