"""Scoring generated answers. Pure functions, no model calls.

Three separate questions, because each catches a different failure:
- CORRECT: does the answer contain the fact the golden set expects ("3 years")? Free string
  check. Catches a confident answer about the wrong rule.
- FAITHFUL: is every claim backed by the sources the model was shown? Needs an LLM judge.
  Catches invented details the citation gate cannot see (a real [1] on a made-up claim).
- ON TOPIC: does it answer THIS question, not a neighbouring one? LLM judge. An answer can be
  perfectly faithful to its source and still be about the wrong requirement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# USD per million tokens (input, output) for prompts up to 100K tokens. Checked 2026-10.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
}

_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                 "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12",
                 "thirty": "30", "sixty": "60", "ninety": "90"}
_WORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b")


def _normalise(text: str) -> str:
    text = text.lower().replace("-", " ").replace("‑", " ")
    text = _WORD.sub(lambda m: _NUMBER_WORDS[m.group(1)], text)
    return re.sub(r"\s+", " ", text)


def _present(phrase: str, have: str) -> bool:
    p = _normalise(phrase).strip()
    return p in have or p.replace(" years", " year") in have


def missing_facts(answer: str, must_contain: list[str]) -> list[str]:
    """Expected phrases absent from the answer. "three years" matches "3 years" and
    "3-year"; case and spacing are ignored. "Inspector General | OIG" is satisfied by
    either alternative (the same fact is often written two ways)."""
    have = _normalise(answer)
    return [p for p in must_contain if not any(_present(alt, have) for alt in p.split("|"))]


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             prices: dict[str, tuple[float, float]] = PRICES) -> float | None:
    """None when the model's price is unknown, rather than a misleading $0."""
    if model not in prices:
        return None
    pin, pout = prices[model]
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        },
        "answers_question": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["claims", "answers_question", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Verdict:
    supported: int
    total: int
    unsupported: tuple[str, ...]
    answers_question: bool
    reason: str

    @property
    def faithful(self) -> bool:
        """All claims supported. An answer with no factual claims counts as faithful."""
        return self.supported == self.total


def parse_verdict(data: dict) -> Verdict:
    claims = data.get("claims")
    if not isinstance(claims, list) or not isinstance(data.get("answers_question"), bool):
        raise ValueError(f"malformed judge verdict: {data!r}")
    unsupported = tuple(str(c.get("claim", "")) for c in claims if not c.get("supported"))
    return Verdict(supported=len(claims) - len(unsupported), total=len(claims),
                   unsupported=unsupported, answers_question=data["answers_question"],
                   reason=str(data.get("reason", "")))


@dataclass
class QuestionResult:
    id: str
    style: str
    should_decline: bool
    declined: bool
    decline_reason: str = ""
    answer: str = ""
    missing: list[str] = field(default_factory=list)
    verdict: Verdict | None = None
    judge_error: str = ""
    cost: float | None = 0.0
    latency_ms: int = 0

    @property
    def correct(self) -> bool:
        """For a should-decline row: it declined. Otherwise: answered with every expected fact."""
        if self.should_decline:
            return self.declined
        return not self.declined and not self.missing


@dataclass(frozen=True)
class AnswerSummary:
    answerable: int
    correct: int
    declined_wrongly: int
    judged: int
    faithful: int
    on_topic: int
    decline_rows: int
    declines_correct: int
    total_cost: float | None
    median_latency_ms: int


def summarise_answers(results: list[QuestionResult]) -> AnswerSummary:
    answerable = [r for r in results if not r.should_decline]
    judged = [r for r in answerable if r.verdict is not None]
    declines = [r for r in results if r.should_decline]
    costs = [r.cost for r in results]
    latencies = sorted(r.latency_ms for r in results) or [0]
    return AnswerSummary(
        answerable=len(answerable),
        correct=sum(r.correct for r in answerable),
        declined_wrongly=sum(r.declined for r in answerable),
        judged=len(judged),
        faithful=sum(r.verdict.faithful for r in judged),
        on_topic=sum(r.verdict.answers_question for r in judged),
        decline_rows=len(declines),
        declines_correct=sum(r.correct for r in declines),
        total_cost=None if any(c is None for c in costs) else sum(costs),
        median_latency_ms=latencies[len(latencies) // 2],
    )
