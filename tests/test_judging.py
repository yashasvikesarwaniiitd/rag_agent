import json
from types import SimpleNamespace

import pytest

from rag.adapters.llms import AnthropicLLM, ScriptedLLM
from rag.application.answer import RagService
from rag.application.judge import JUDGE_SYSTEM, build_judge_prompt, evaluate_answers, judge_answer
from rag.domain.judging import (JUDGE_SCHEMA, QuestionResult, cost_usd, missing_facts,
                                parse_verdict, summarise_answers)
from rag.domain.models import Answer, Chunk, Hit


# ---------------------------------------------------------------- correctness (free)

def test_number_words_and_digits_match():
    assert missing_facts("Recredential every three years [1].", ["3 years"]) == []
    assert missing_facts("on a 3-year cycle [1]", ["3 years"]) == []


def test_wrong_rule_is_caught():
    # The real failure seen in the UI: "re-checked" answered with the 5-year enrollment rule.
    assert missing_facts("Revalidate every five years [1].", ["3 years"]) == ["3 years"]


def test_case_and_spacing_are_ignored():
    assert missing_facts("Must hold   LTSS and Mental Health", ["mental health", "LTSS"]) == []


# ---------------------------------------------------------------- cost (free)

def test_cost_uses_per_million_prices():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("claude-haiku-5-5", 2000, 500) == pytest.approx(0.00045)


def test_unknown_model_cost_is_none_not_zero():
    assert cost_usd("some-other-model", 1000, 1000) is None


# ---------------------------------------------------------------- verdicts

def test_parse_verdict_counts_unsupported_claims():
    v = parse_verdict({"claims": [{"claim": "a", "supported": True},
                                  {"claim": "b", "supported": False}],
                       "answers_question": True, "reason": "b is invented"})
    assert (v.supported, v.total, v.unsupported, v.faithful) == (1, 2, ("b",), False)


def test_malformed_verdict_raises():
    with pytest.raises(ValueError):
        parse_verdict({"claims": "nope", "answers_question": True})


def test_schema_is_strict():
    assert JUDGE_SCHEMA["additionalProperties"] is False
    assert set(JUDGE_SCHEMA["required"]) == {"claims", "answers_question", "reason"}


# ---------------------------------------------------------------- judge call

def _answer():
    chunk = Chunk("c", "doc", "ch6.pdf", 13, "Recredential at least every 3 years.")
    return Answer(text="Every 3 years [1].", hits=[Hit(chunk, 1.0)])


def test_judge_prompt_holds_sources_question_and_answer():
    prompt = build_judge_prompt("How often?", _answer())
    assert '<source id="1" document="ch6.pdf" page="13">' in prompt
    assert "Question: How often?" in prompt and prompt.endswith("Every 3 years [1].")


def test_judge_answer_requests_json_and_parses_it():
    llm = ScriptedLLM([json.dumps({"claims": [{"claim": "every 3 years", "supported": True}],
                                   "answers_question": True, "reason": "ok"})])
    verdict, _ = judge_answer(llm, "How often?", _answer())
    assert verdict.faithful and verdict.answers_question
    assert llm.calls[0]["json_schema"] == JUDGE_SCHEMA and llm.calls[0]["system"] == JUDGE_SYSTEM


class FakeClient:
    def __init__(self, response):
        self.response, self.kwargs, self.messages = response, None, self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _resp(text="{}", stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, model="claude-haiku-5-5",
                           content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def test_adapter_sends_schema_and_effort_in_output_config():
    client = FakeClient(_resp())
    AnthropicLLM(client=client, effort="low").complete(system="s", user="u",
                                                       json_schema={"type": "object"})
    assert client.kwargs["output_config"] == {
        "effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}


def test_adapter_sends_no_output_config_by_default():
    client = FakeClient(_resp())
    AnthropicLLM(client=client).complete(system="s", user="u")
    assert "output_config" not in client.kwargs


# ---------------------------------------------------------------- evaluation loop ($0)

def test_truncated_answer_is_declined_not_shown(indexed_store):
    from rag.ports import LLMResult

    class Truncating:
        def complete(self, **kw):
            return LLMResult(text="Every 36 mon [1", stop_reason="max_tokens")

    embedder, store = indexed_store
    answer = RagService(embedder=embedder, store=store, llm=Truncating()).ask("recredentialing?")
    assert answer.refused and "truncated" in answer.reason


def test_evaluate_answers_scores_correct_wrong_and_declines(indexed_store):
    embedder, store = indexed_store
    replies = {"license": "Every five years [1].", "recredential": "At least every 36 months [1]."}
    llm = ScriptedLLM(lambda s, u: next((v for k, v in replies.items()
                                         if k in u.rsplit("Question:", 1)[-1].lower()),
                                        "NOT_FOUND"))
    def fake_judge(system, user):
        good = "36 months" in user.rsplit("Answer to grade:", 1)[-1]
        return json.dumps({"claims": [{"claim": "x", "supported": good}],
                           "answers_question": good, "reason": "r"})

    judge = ScriptedLLM(fake_judge)
    # Hashing vectors: on-topic questions score ~0.2, "capital of France" 0.0.
    svc = RagService(embedder=embedder, store=store, llm=llm, min_score=0.1)
    rows = [
        {"id": "ok", "style": "policy", "question": "How often are providers recredentialed?",
         "must_contain": ["36 months"], "should_decline": False},
        {"id": "wrong", "style": "plain", "question": "How often is the state license verified?",
         "must_contain": ["3 years"], "should_decline": False},
        {"id": "off", "style": "decline", "question": "What is the capital of France?",
         "must_contain": [], "should_decline": True},
    ]
    results = {r.id: r for r in evaluate_answers(rows, svc, judge, answer_is_fake=True,
                                                 judge_is_fake=True)}
    assert results["ok"].correct and results["ok"].verdict.faithful
    assert not results["wrong"].correct and results["wrong"].missing == ["3 years"]
    assert results["wrong"].verdict.answers_question is False
    assert results["off"].correct and results["off"].verdict is None  # never judged
    s = summarise_answers(list(results.values()))
    assert (s.correct, s.answerable, s.faithful, s.judged, s.declines_correct) == (1, 2, 1, 2, 1)
    assert s.total_cost == 0.0


def test_a_bad_verdict_is_recorded_without_losing_the_run(indexed_store):
    embedder, store = indexed_store
    svc = RagService(embedder=embedder, store=store, llm=ScriptedLLM(["Every 36 months [1]."]))
    rows = [{"id": "q", "style": "policy", "question": "recredentialing?",
             "must_contain": [], "should_decline": False}]
    [res] = evaluate_answers(rows, svc, ScriptedLLM(["not json"]), answer_is_fake=True,
                             judge_is_fake=True)
    assert res.verdict is None and res.judge_error


def _paid_service(indexed_store, cost_per_call_tokens=(100_000, 50_000)):
    """A 'paid' fake: reports real-looking token counts priced as Haiku 5.5."""
    from rag.ports import LLMResult

    class Billed:
        def __init__(self):
            self.calls = 0

        def complete(self, **kw):
            self.calls += 1
            return LLMResult(text="Every 36 months [1].", input_tokens=cost_per_call_tokens[0],
                             output_tokens=cost_per_call_tokens[1], model="claude-haiku-5-5")

    embedder, store = indexed_store
    llm = Billed()
    return RagService(embedder=embedder, store=store, llm=llm), llm


def test_spending_cap_stops_the_run(indexed_store):
    # Each answer costs 100k*0.10/1M + 50k*0.50/1M = $0.035.
    svc, llm = _paid_service(indexed_store)
    rows = [{"id": f"q{i}", "style": "policy", "question": "recredentialing?",
             "must_contain": [], "should_decline": False} for i in range(10)]
    results = evaluate_answers(rows, svc, None, max_cost=0.05)
    assert len(results) == 2 and llm.calls == 2  # $0.035 < cap, $0.070 >= cap -> stop
    assert summarise_answers(results).total_cost == pytest.approx(0.07)


def test_no_cap_runs_everything(indexed_store):
    svc, llm = _paid_service(indexed_store)
    rows = [{"id": f"q{i}", "style": "policy", "question": "recredentialing?",
             "must_contain": [], "should_decline": False} for i in range(4)]
    assert len(evaluate_answers(rows, svc, None)) == 4


def test_unknown_price_stops_a_capped_run(indexed_store):
    from rag.ports import LLMResult

    class Unpriced:
        def complete(self, **kw):
            return LLMResult(text="x [1]", input_tokens=10, output_tokens=10, model="mystery")

    embedder, store = indexed_store
    svc = RagService(embedder=embedder, store=store, llm=Unpriced())
    rows = [{"id": f"q{i}", "style": "policy", "question": "recredentialing?",
             "must_contain": [], "should_decline": False} for i in range(3)]
    assert len(evaluate_answers(rows, svc, None, max_cost=1.0)) == 1


def test_summary_cost_is_unknown_if_any_price_is_unknown():
    rs = [QuestionResult("a", "policy", False, False, cost=0.01),
          QuestionResult("b", "policy", False, False, cost=None)]
    assert summarise_answers(rs).total_cost is None
