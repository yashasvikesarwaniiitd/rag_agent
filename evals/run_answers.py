"""Generate answers for the golden set and grade them.

    uv run python evals/run_answers.py --dry-run              # $0: estimate the cost, no calls
    uv run python evals/run_answers.py --fake                 # $0: whole loop with fake models
    uv run python evals/run_answers.py --limit 3              # PAID: first 3 questions only
    uv run python evals/run_answers.py --save v0.2-haiku      # PAID: all 22, write a report

Paid runs need ANTHROPIC_API_KEY in .env. Answers and judge both use RAG_MODEL
(default claude-haiku-5-5) at effort "low" unless --effort says otherwise, and stop at
--max-cost (default $0.05) - the cost is read from each response's real token counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.adapters.llms import AnthropicLLM, ExtractiveLLM, ScriptedLLM  # noqa: E402
from rag.application.judge import JUDGE_SYSTEM, build_judge_prompt, evaluate_answers  # noqa: E402
from rag.application.prompts import SYSTEM, build_user_prompt  # noqa: E402
from rag.config import (Settings, build_embedder, build_reranker, build_service,  # noqa: E402
                        build_store)
from rag.domain.judging import PRICES, summarise_answers  # noqa: E402
from rag.domain.models import Answer  # noqa: E402

GOLDEN = ROOT / "evals" / "golden.jsonl"
CHARS_PER_TOKEN = 4  # rough for English; the real count comes back on every paid call
# Haiku 5.5 always thinks; thinking is billed as output. Low/high guesses for one call.
OUTPUT_TOKENS_RANGE = (300, 1500)


def load_golden() -> list[dict]:
    return [json.loads(x) for x in GOLDEN.read_text(encoding="utf-8").splitlines() if x.strip()]


def dry_run(rows, service, model: str) -> str:
    pin, pout = PRICES[model]
    lo_out, hi_out = OUTPUT_TOKENS_RANGE
    calls, input_tokens, skipped = 0, 0, 0
    for row in rows:
        hits, best = service.search(row["question"])
        if best < service.min_score:
            skipped += 1  # declined by the off-topic cutoff before any model call
            continue
        answer_prompt = SYSTEM + build_user_prompt(row["question"], hits)
        fake_answer = Answer(text="x" * 600, hits=hits)  # ~150-token answer to be judged
        judge_prompt = JUDGE_SYSTEM + build_judge_prompt(row["question"], fake_answer)
        input_tokens += (len(answer_prompt) + len(judge_prompt)) // CHARS_PER_TOKEN
        calls += 2
    lo = (input_tokens * pin + calls * lo_out * pout) / 1e6
    hi = (input_tokens * pin + calls * hi_out * pout) / 1e6
    return (f"Dry run, {model}: {len(rows)} questions, {skipped} declined before any call, "
            f"{calls} model calls (answer + judge).\n"
            f"  input  ~{input_tokens:,} tokens (measured from the real prompts)\n"
            f"  output {calls * lo_out:,}-{calls * hi_out:,} tokens (assumed "
            f"{lo_out}-{hi_out} per call, thinking included)\n"
            f"  estimated cost ${lo:.3f} - ${hi:.3f} per full run "
            f"(~Rs {lo * 88:.1f} - {hi * 88:.1f})")


def report(results, label: str) -> str:
    s = summarise_answers(results)
    cost = "unknown (price missing)" if s.total_cost is None else f"${s.total_cost:.4f}"
    lines = [f"# Answer eval - {label} - {date.today().isoformat()}", "",
             "| Measure | Result |", "|---|---|",
             f"| Correct (expected fact present AND on topic) | {s.correct}/{s.answerable} |",
             f"| Declined a question it should answer | {s.declined_wrongly}/{s.answerable} |",
             f"| Faithful (every claim supported, per judge) | {s.faithful}/{s.judged} judged |",
             f"| On topic (per judge) | {s.on_topic}/{s.judged} judged |",
             f"| Declined when it should | {s.declines_correct}/{s.decline_rows} |",
             f"| Total cost | {cost} |",
             f"| Median latency | {s.median_latency_ms} ms |", "",
             "| Question | Style | Result | Missing fact | Faithful | On topic | Note |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        v = r.verdict
        outcome = ("declined" if r.declined else "answered") + (" ✓" if r.correct else " ✗")
        faithful = f"{v.supported}/{v.total}" if v else ("error" if r.judge_error else "-")
        topic = ("yes" if v.answers_question else "**no**") if v else "-"
        note = (r.decline_reason or r.judge_error or (v.reason if v and v.reason != "ok" else "")
                or (("unsupported: " + "; ".join(v.unsupported)) if v and v.unsupported else ""))
        lines.append(f"| {r.id} | {r.style} | {outcome} | {', '.join(r.missing) or '-'} | "
                     f"{faithful} | {topic} | {note[:140].replace('|', '/')} |")
    lines += ["", "Faithful/on-topic come from a Haiku judge grading Haiku's answers, which "
              "tends to be lenient; read the unsupported claims, not just the counts."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="estimate cost, make no calls")
    mode.add_argument("--fake", action="store_true", help="extractive answers + fake judge, $0")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="low",
                        help="Haiku effort for answers and judge. Thinking is billed as "
                             "output, so 'low' (default) is the cheap setting")
    parser.add_argument("--max-cost", type=float, default=0.05, metavar="USD",
                        help="stop asking questions once this much has been spent "
                             "(default $0.05; overshoot is at most one question)")
    parser.add_argument("--no-judge", action="store_true", help="skip the paid judge call")
    parser.add_argument("--save", metavar="NAME", help="write evals/results/answers-NAME.md/.jsonl")
    args = parser.parse_args()
    # Windows consoles default to cp1252, which cannot print the ✓/✗ marks.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    s = Settings.from_env()
    rows = load_golden()[: args.limit] if args.limit else load_golden()
    embedder, store = build_embedder(s), build_store(s)

    if args.dry_run:
        service = build_service(s, embedder, store, ExtractiveLLM(), reranker=build_reranker(s))
        print(dry_run(rows, service, s.model))
        return 0

    if args.fake:
        answer_llm = ExtractiveLLM()
        judge = ScriptedLLM(lambda system, user: json.dumps(
            {"claims": [{"claim": "fake", "supported": True}], "answers_question": True,
             "reason": "fake judge"}))
        label = "fake models ($0)"
    else:
        answer_llm = AnthropicLLM(model=s.model, effort=args.effort)
        judge = AnthropicLLM(model=s.model, effort=args.effort)
        label = f"{s.model}" + (f", effort {args.effort}" if args.effort else "")
    service = build_service(s, embedder, store, answer_llm, reranker=build_reranker(s))

    def progress(r):
        mark = "✓" if r.correct else "✗"
        print(f"  {mark} {r.id:34} {'declined' if r.declined else 'answered'}"
              f"{'  cost $%.4f' % r.cost if r.cost else ''}", flush=True)

    results = evaluate_answers(rows, service, None if args.no_judge else judge,
                               answer_is_fake=args.fake, judge_is_fake=args.fake,
                               max_cost=None if args.fake else args.max_cost,
                               on_result=progress)
    text = report(results, label)
    if len(results) < len(rows):
        text += (f"\n\n**Stopped at the ${args.max_cost:.2f} spending cap after "
                 f"{len(results)}/{len(rows)} questions.** Raise --max-cost to run the rest.")
    print("\n" + text)
    if args.save:
        out = ROOT / "evals" / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"answers-{args.save}.md").write_text(text + "\n", encoding="utf-8")
        with open(out / f"answers-{args.save}.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        print(f"\nSaved evals/results/answers-{args.save}.md and .jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
