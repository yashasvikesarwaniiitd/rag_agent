"""The citation gate: an answer reaches the user only if every claim it cites points at a
source we actually retrieved. The prompt asks for this, but prompts are not a safety
mechanism on their own - this check is.

v0 rules:
- the model may answer exactly NOT_FOUND, which is an honest "the sources don't cover it";
- otherwise the answer must carry at least one [n] marker;
- every marker must be in range 1..n_sources (an out-of-range number is a made-up source).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NOT_FOUND = "NOT_FOUND"
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class GateResult:
    ok: bool
    not_found: bool = False
    cited: frozenset[int] = field(default_factory=frozenset)
    invalid: frozenset[int] = field(default_factory=frozenset)
    reason: str = ""


def cited_numbers(text: str) -> list[int]:
    """All numbers inside [n] or [n, m] markers, in order of first appearance."""
    seen: list[int] = []
    for group in _MARKER.findall(text):
        for part in group.split(","):
            n = int(part.strip())
            if n not in seen:
                seen.append(n)
    return seen


def check_answer(text: str, n_sources: int) -> GateResult:
    stripped = text.strip()
    if stripped.strip(".").upper() == NOT_FOUND:
        return GateResult(ok=True, not_found=True, reason="sources do not cover the question")
    if not stripped:
        return GateResult(ok=False, reason="empty answer")

    numbers = cited_numbers(stripped)
    if not numbers:
        return GateResult(ok=False, reason="answer has no citations")
    valid = frozenset(n for n in numbers if 1 <= n <= n_sources)
    invalid = frozenset(n for n in numbers if n not in valid)
    if invalid:
        return GateResult(ok=False, cited=valid, invalid=invalid,
                          reason=f"answer cites sources that were not retrieved: {sorted(invalid)}")
    return GateResult(ok=True, cited=valid)
