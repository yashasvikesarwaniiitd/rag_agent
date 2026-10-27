"""Score retrieval against evals/golden.jsonl. Free: no model calls, only local search.

    uv run python evals/run_retrieval.py                    # all modes, print table
    uv run python evals/run_retrieval.py --save baseline    # also write evals/results/baseline.md

Decline rows are scored on the off-topic cutoff alone (best vector match < RAG_MIN_SCORE),
because that is the only decline that happens before a model call. A decline that needs
judgement ("should we approve Dr X?") is expected to FAIL here until the router exists -
it is in the set so that gap stays visible instead of forgotten.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.adapters.llms import ExtractiveLLM  # noqa: E402
from rag.config import Settings, build_embedder, build_store  # noqa: E402
from rag.application.answer import RagService  # noqa: E402
from rag.domain.scoring import Expected, first_relevant_rank, summarise  # noqa: E402

GOLDEN = ROOT / "evals" / "golden.jsonl"
MODES = ["vector", "hybrid"]


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def evaluate(rows: list[dict], services: dict[str, RagService], k: int):
    per_q: dict[str, dict[str, int | None]] = {}
    declines: dict[str, dict[str, bool]] = {}
    for row in rows:
        for mode, svc in services.items():
            hits, best = svc.search(row["question"])
            if row["should_decline"]:
                declines.setdefault(row["id"], {})[mode] = best < svc.min_score
                continue
            expected = [Expected(**e) for e in row["expected"]]
            per_q.setdefault(row["id"], {})[mode] = first_relevant_rank(
                [h.chunk for h in hits[:k]], expected)
    return per_q, declines


def report(rows, per_q, declines, modes, k) -> str:
    style = {r["id"]: r["style"] for r in rows}
    lines = [f"# Retrieval eval - {date.today().isoformat()}", "",
             f"{len(per_q)} answerable questions, top-{k}. Rank = position of the first chunk "
             "on an expected page; '-' = not in the top k.", "",
             "| Mode | Found (all) | MRR | Found (policy wording) | Found (plain English) |",
             "|---|---|---|---|---|"]
    for mode in modes:
        overall = summarise([per_q[q][mode] for q in per_q], k)
        by = {s: summarise([per_q[q][mode] for q in per_q if style[q] == s], k)
              for s in ("policy", "plain")}
        lines.append(f"| {mode} | {overall.hits}/{overall.questions} | {overall.mrr:.2f} | "
                     f"{by['policy'].hits}/{by['policy'].questions} | "
                     f"{by['plain'].hits}/{by['plain'].questions} |")
    lines += ["", "| Question | Style | " + " | ".join(modes) + " |",
              "|---|---|" + "---|" * len(modes)]
    for q, ranks in per_q.items():
        cells = " | ".join(str(ranks[m]) if ranks[m] else "-" for m in modes)
        lines.append(f"| {q} | {style[q]} | {cells} |")
    lines += ["", "| Should decline | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for q, res in declines.items():
        lines.append(f"| {q} | " + " | ".join("declined" if res[m] else "**answered**"
                                              for m in modes) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="NAME", help="write evals/results/NAME.md")
    parser.add_argument("--modes", nargs="+", default=MODES)
    args = parser.parse_args()

    s = Settings.from_env()
    embedder, store = build_embedder(s), build_store(s)
    services = {m: RagService(embedder=embedder, store=store, llm=ExtractiveLLM(),
                              top_k=s.top_k, retrieval=m, candidates=s.candidates,
                              min_score=s.min_score) for m in args.modes}
    rows = load_golden()
    per_q, declines = evaluate(rows, services, s.top_k)
    text = report(rows, per_q, declines, args.modes, s.top_k)
    print(text)
    if args.save:
        out = ROOT / "evals" / "results" / f"{args.save}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
