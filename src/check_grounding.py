"""
Automated check that the agent never invents a number.

    python -m src.check_grounding

Every dollar figure in an answer is extracted and matched against the tool
output that produced it. Anything the model wrote that pandas did not compute is
a hallucination and gets reported.

This exists because a hallucination actually happened. Asked "how much am I
paying for subscriptions?", an earlier version replied "$131.65 per month" -- a
figure that appeared nowhere in the data. The model had decided which recurring
charges counted as subscriptions and added them up itself, incorrectly. The
system prompt at the time said "use only numbers that appear in DATA", which the
model followed loosely enough to sum two of them together.

The fix was an explicit ban on arithmetic in ANSWER_SYSTEM. This script is how
we know the fix holds, and how you would notice if a prompt edit broke it again.
For a finance tool, a confidently wrong number is worse than no answer.
"""

from __future__ import annotations

import re

from src.agent import DEMO_QUESTIONS, ask
from src.features import load_transactions

MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")


def _variants(values) -> set[str]:
    """Same number written different ways: 1,450 / 1450 / 1450.00 / 1450.0."""
    out: set[str] = set()
    for value in values:
        clean = value.replace(",", "")
        out.add(clean)
        try:
            number = float(clean)
        except ValueError:
            continue
        out.update({f"{number:.2f}", f"{number:.1f}", f"{number:g}"})
    return out


def check(questions: list[str] | None = None) -> int:
    df = load_transactions()
    questions = questions or DEMO_QUESTIONS
    quoted = ungrounded = 0

    for question in questions:
        out = ask(df, question)
        said = MONEY.findall(out["answer"])
        available = _variants(MONEY.findall(out["data_text"]))
        bad = [s for s in said if not (_variants([s]) & available)]

        quoted += len(said)
        ungrounded += len(bad)

        print(f"[{'OK  ' if not bad else 'FAIL'}] {question}")
        print(f"        tool: {out['tool']:20s} figures: {len(said)}  "
              f"ungrounded: {len(bad)}")
        if bad:
            print(f"        NOT IN DATA: {bad}")
            print(f"        answer: {out['answer'][:200]}")

    print("\n" + "=" * 70)
    print(f"Dollar figures quoted : {quoted}")
    print(f"Ungrounded            : {ungrounded}")
    print("RESULT:", "PASS -- every figure traced back to pandas"
          if ungrounded == 0 else f"FAIL -- {ungrounded} invented")
    return ungrounded


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
