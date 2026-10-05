"""
The agent: four tools, a router, and a grounded answer.

    python -m src.agent            # runs a set of demo questions

How it works
------------
An "AI agent" sounds mysterious. This one is three steps:

1. **Route.** Show the language model the user's question and a list of the four
   tools. It replies with JSON naming one tool and its arguments. It does not
   see any transaction data at this point -- it is choosing, not answering.

2. **Execute.** We run the chosen pandas function. This is ordinary code with no
   model involved, so the numbers are simply correct.

3. **Phrase.** Show the model the question again plus the computed results, and
   ask it to write the reply. It is instructed to use only the numbers in front
   of it.

The split matters. The model never calculates anything; it picks a function and
writes prose about the output. That is the difference between an assistant that
occasionally invents a plausible dollar figure and one that cannot.

Every step degrades. No API key, quota exhausted, malformed JSON, hallucinated
tool name -- each falls back to keyword routing and a template answer built
straight from the DataFrame. The demo never dies because a free quota reset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd

from src import analytics
from src.features import (add_time_features, load_transactions,
                          normalise_merchant, spending_only)
from src.llm import (LLMUnavailable, complete, complete_stream,
                     is_available)
from src.train_fraud_detector import (FRAUD_FEATURES, add_fraud_features,
                                      build_isolation_forest, explain_anomaly)

CATEGORIZER_PATH = Path("models/categorizer.pkl")

# Words a `Type` column uses for money coming in and money going out. Reading
# this beats inferring from the sign: plenty of real exports list every amount
# as positive and record the direction in a separate column, and guessing from
# the sign there turns every expense into income.
_CREDIT_TYPES = {"income", "salary", "deposit", "credit", "refund",
                 "cashback", "in", "received"}
_DEBIT_TYPES = {"expense", "debit", "purchase", "payment", "withdrawal",
                "atm", "transfer", "out", "sent", "spend"}

# Moving your own money between your own accounts is not spending, and neither
# is taking cash out or paying off your own card. Counting them inflates every
# total and makes the app disagree with the statement it was given.
_MOVEMENT_TERMS = {"transfer", "atm", "atm withdrawal", "cash withdrawal",
                   "credit card payment", "card payment", "self transfer",
                   "savings", "saved to goals", "goal", "internal transfer"}


def _direction_from_type(value) -> str | None:
    word = str(value).strip().lower()
    if word in _CREDIT_TYPES:
        return "credit"
    if word in _DEBIT_TYPES:
        return "debit"
    return None


def _movement_flag(df: pd.DataFrame) -> pd.Series:
    """
    1 where a row moves money without spending it.

    Only the `Type` column is consulted, deliberately. Reading `Category` too
    looked sensible and was wrong: in a real export, a *category* of "Transfer"
    usually means money sent to another person, which is spending, while the
    export's own summary reserves "transfers" for movements between your own
    accounts. Excluding on category removed EGP 7,014 of genuine spending and
    put the app EGP 7,014 adrift from the statement it had just been handed.

    Type describes the mechanism; category describes the purpose. Only the
    mechanism tells you whether the money actually left.
    """
    if "type" not in df.columns:
        return pd.Series(0, index=df.index)
    words = df["type"].astype(str).str.strip().str.lower()
    return words.isin(_MOVEMENT_TERMS).astype(int)


# ==========================================================================
# Preparing an arbitrary CSV
# ==========================================================================

def prepare_dataframe(df: pd.DataFrame,
                      recategorise: bool = False) -> pd.DataFrame:
    """
    Make any transaction CSV usable.

    This is where the categoriser finally earns its keep. Our own generated data
    ships with a `category` column, but a real bank export has no such thing --
    it has a date, a description and an amount. When the column is missing we
    predict it, which is the entire point of having trained the model.

    `recategorise=True` throws away a category column the file already has and
    predicts every one from scratch. Some exports arrive pre-labelled by whatever
    app produced them, and keeping those labels means the classifier never runs.
    """
    df = df.copy()

    # Lower-case every header. Real exports are wildly inconsistent -- "Date",
    # "DATE", "Description ", "Amount" -- and matching case-sensitively means
    # the lookups below silently miss, then fail much later with a confusing
    # KeyError somewhere deep in the model.
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "timestamp" not in df.columns:
        for alt in ("date", "datetime", "time", "posted", "transaction date"):
            if alt in df.columns:
                df = df.rename(columns={alt: "timestamp"})
                break
    if "timestamp" not in df.columns:
        raise ValueError("No date column found. Expected one of: timestamp, "
                         f"date, datetime, time. Got: {', '.join(df.columns)}")
    if "amount" not in df.columns:
        raise ValueError("No amount column found. "
                         f"Got: {', '.join(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for alt in ("memo", "details", "narrative", "payee"):
        if "description" not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: "description"})
    if "description" not in df.columns and "merchant" in df.columns:
        df["description"] = df["merchant"]
    if "channel" not in df.columns:
        df["channel"] = "card_present"

    if "direction" not in df.columns:
        if "type" in df.columns:
            # Trust an explicit Type column, and fall back to the sign only for
            # rows whose type we do not recognise.
            df["direction"] = df["type"].map(_direction_from_type)
            unknown = df["direction"].isna()
            if unknown.any():
                df.loc[unknown, "direction"] = [
                    "credit" if a > 0 else "debit"
                    for a in df.loc[unknown, "amount"]]
        else:
            # Convention in most exports: negative means money left the account.
            df["direction"] = ["credit" if a > 0 else "debit"
                               for a in df["amount"]]
    df["amount"] = df["amount"].abs()

    df["is_movement"] = _movement_flag(df)
    if "city" not in df.columns:
        df["city"] = "Unknown"
    if "country" not in df.columns:
        df["country"] = "Unknown"
    if "merchant" not in df.columns:
        # Not the raw descriptor. Every bill carries a fresh reference number,
        # so grouping on the raw string finds no repeats and subscription
        # detection collapses. See normalise_merchant.
        df["merchant"] = df["description"].map(normalise_merchant)

    df = add_time_features(df)

    if recategorise and "category" in df.columns:
        df = df.drop(columns=["category"])

    if "category" not in df.columns:
        if not CATEGORIZER_PATH.exists():
            raise FileNotFoundError(
                "No category column and no trained model. "
                "Run: python -m src.train_categorizer")
        model = joblib.load(CATEGORIZER_PATH)
        debit = df["direction"] == "debit"
        df.loc[:, "category"] = "Income"
        df.loc[debit, "category"] = model.predict(df[debit])

    return df


# ==========================================================================
# The four tools
# ==========================================================================

def tool_spending_summary(df: pd.DataFrame, month: str | None = None) -> dict:
    """Category breakdown, totals and fixed commitments."""
    month = month or analytics.latest_month(df)
    table = analytics.category_breakdown(df, month)
    monthly = analytics.monthly_summary(df)
    subs = analytics.recurring_charges(df)
    row = monthly.loc[month] if month in monthly.index else None

    lines = [f"Month: {month}"]
    if row is not None:
        summary = (f"Income EGP {row['income']:,.2f} | Spending "
                   f"EGP {row['spending']:,.2f} | Net EGP {row['net']:,.2f}")
        # No income recorded means no savings rate. Printing "nan%" at a user is
        # worse than saying nothing.
        if pd.notna(row["savings_rate"]):
            summary += f" | Savings rate {row['savings_rate']:.1f}%"
        lines.append(summary)
    lines.append("\nSpending by category:")
    for cat, r in table.iterrows():
        lines.append(f"  {cat}: EGP {r['total']:,.2f} across "
                     f"{int(r['transactions'])} transactions "
                     f"({r['share_pct']:.1f}% of spending)")
    # List the subscriptions individually, not just their total. An earlier
    # version reported only the sum, and when asked "how much am I paying for
    # subscriptions?" the model correctly refused to answer -- it could see a
    # figure for "fixed commitments" but had no way to know what was in it.
    # Tools should hand over the detail, not a number the model has to trust.
    lines.append(f"\nRecurring monthly commitments -- "
                 f"EGP {subs['monthly_cost'].sum():,.2f} total across "
                 f"{len(subs)} charges:")
    for merchant, r in subs.iterrows():
        lines.append(f"  {merchant}: EGP {r['monthly_cost']:,.2f} per month")

    return {"tool": "spending_summary",
            "headline": f"Spending breakdown for {month}",
            "data_text": "\n".join(lines),
            "table": table}


def tool_find_overspending(df: pd.DataFrame, month: str | None = None,
                           lookback: int = 3) -> dict:
    """Categories running above this account's own recent norm."""
    month = month or analytics.latest_month(df)
    table = analytics.overspending_signals(df, month, lookback=lookback)
    over = table[table["over_by"] > 0]
    under = table[table["over_by"] < 0]

    lines = [f"Month analysed: {month}",
             f"Baseline: average of the previous {lookback} months",
             "\nOver the usual:"]
    for cat, r in over.iterrows():
        pct = "" if pd.isna(r["over_by_pct"]) else f" (+{r['over_by_pct']:.0f}%)"
        lines.append(f"  {cat}: EGP {r['this_month']:,.2f} vs usual "
                     f"EGP {r['usual']:,.2f} -> over by EGP {r['over_by']:,.2f}{pct}")
    if not under.empty:
        lines.append("\nBelow the usual:")
        for cat, r in under.iterrows():
            lines.append(f"  {cat}: EGP {r['this_month']:,.2f} vs usual "
                         f"EGP {r['usual']:,.2f} -> under by "
                         f"EGP {abs(r['over_by']):,.2f}")

    return {"tool": "find_overspending",
            "headline": f"Overspending analysis for {month}",
            "data_text": "\n".join(lines),
            "table": table}


def tool_compare_months(df: pd.DataFrame, current: str | None = None,
                        previous: str | None = None) -> dict:
    """Category-level difference between two months."""
    current = current or analytics.latest_month(df)
    table = analytics.compare_months(df, current, previous)
    monthly = analytics.monthly_summary(df)
    prev_col = [c for c in table.columns if c not in
                (current, "change", "change_pct")][0]

    # Every figure is tagged with the month it belongs to. Without the tags the
    # earlier version emitted bare "EGP 52,831.64 -> EGP 48,462.30" under a
    # header reading "Comparing 2026-06 against 2026-05", and the model matched
    # the numbers to the months in the order the header named them -- backwards.
    # If a number can be misread, eventually it will be.
    lines = [f"Earlier month: {prev_col}. Later month: {current}.",
             f"All figures below are written as {prev_col} first, "
             f"then {current}."]
    if current in monthly.index and prev_col in monthly.index:
        a, b = monthly.loc[prev_col], monthly.loc[current]
        lines.append(f"\nTotal spending: {prev_col} EGP {a['spending']:,.2f} -> "
                     f"{current} EGP {b['spending']:,.2f} "
                     f"(change EGP {b['spending'] - a['spending']:+,.2f})")
    lines.append("\nBy category, largest change first:")
    for cat, r in table.iterrows():
        pct = "" if pd.isna(r["change_pct"]) else f" ({r['change_pct']:+.0f}%)"
        lines.append(f"  {cat}: {prev_col} EGP {r[prev_col]:,.2f} -> "
                     f"{current} EGP {r[current]:,.2f} "
                     f"-> change EGP {r['change']:+,.2f}{pct}")

    return {"tool": "compare_months",
            "headline": f"{current} vs {prev_col}",
            "data_text": "\n".join(lines),
            "table": table}


def tool_detect_suspicious(df: pd.DataFrame, month: str | None = None,
                           top_n: int = 8) -> dict:
    """
    Flag unusual transactions and explain each one.

    The Isolation Forest is refitted on whatever data it is given rather than
    loaded from disk. Anomaly detection is relative -- "unusual" means unusual
    *for this account*. A saved model carries our fictional Austin engineer's
    habits, which say nothing about the next person who uploads a CSV.
    """
    work = add_fraud_features(spending_only(df))
    if month:
        work = work[work["month"] == month]
    if len(work) < 30:
        return {"tool": "detect_suspicious",
                "headline": "Not enough data",
                "data_text": f"Only {len(work)} transactions available; "
                             "anomaly detection needs at least 30.",
                "table": pd.DataFrame()}

    pipe = build_isolation_forest()
    pipe.fit(work[FRAUD_FEATURES])
    work = work.assign(
        anomaly_score=-pipe.decision_function(work[FRAUD_FEATURES]),
        flagged=(pipe.predict(work[FRAUD_FEATURES]) == -1).astype(int),
    )

    flagged = work.nlargest(top_n, "anomaly_score")
    lines = [f"Scanned {len(work):,} transactions, flagged "
             f"{int(work['flagged'].sum())} as unusual.",
             f"\nMost suspicious {len(flagged)}:"]
    for _, r in flagged.iterrows():
        lines.append(f"\n  {r['timestamp']:%Y-%m-%d %H:%M}  "
                     f"EGP {r['amount']:,.2f}  {r['description'][:40]}")
        for reason in explain_anomaly(r):
            lines.append(f"      reason: {reason}")

    cols = ["timestamp", "description", "amount", "category", "anomaly_score"]
    if "is_fraud" in flagged.columns:
        cols.append("is_fraud")

    table = flagged[cols].copy()
    numeric = table.select_dtypes("number").columns      # not the timestamp
    table[numeric] = table[numeric].round(3)

    return {"tool": "detect_suspicious",
            "headline": f"{int(work['flagged'].sum())} unusual transactions",
            "data_text": "\n".join(lines),
            "table": table}


TOOLS = {
    "spending_summary": tool_spending_summary,
    "find_overspending": tool_find_overspending,
    "compare_months": tool_compare_months,
    "detect_suspicious": tool_detect_suspicious,
}

TOOL_SPECS = [
    {"name": "spending_summary",
     "use_when": "the user asks what they spent, on what, how much in total, "
                 "their biggest category, or about subscriptions and fixed bills",
     "args": {"month": "optional, format YYYY-MM, defaults to the latest month"}},
    {"name": "find_overspending",
     "use_when": "the user asks where they are overspending, what is too high, "
                 "where to cut back, or how to save money",
     "args": {"month": "optional, YYYY-MM",
              "lookback": "optional integer, months of history for the baseline"}},
    {"name": "compare_months",
     "use_when": "the user compares two time periods, e.g. this month vs last "
                 "month, or asks what changed",
     "args": {"current": "optional, YYYY-MM", "previous": "optional, YYYY-MM"}},
    {"name": "detect_suspicious",
     "use_when": "the user asks about fraud, suspicious, unusual, strange or "
                 "unrecognised transactions, or whether their card was misused",
     "args": {"month": "optional, YYYY-MM",
              "top_n": "optional integer, how many to return"}},
]


# ==========================================================================
# Routing
# ==========================================================================

ROUTER_SYSTEM = """You route personal-finance questions to exactly one tool.

Reply with JSON only: {"tool": "<name>", "args": {...}, "reason": "<short>"}

Rules:
- Pick exactly one tool from the list.
- Include an argument only if the question clearly specifies it. Omit otherwise;
  the defaults are sensible.
- Months use YYYY-MM. Never invent a month that is not in the available list.
- If the question is vague, choose spending_summary."""

MONTH_NAMES = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

RULE_PATTERNS = [
    ("detect_suspicious", r"fraud|suspicious|unusual|strange|weird|anomal|"
                          r"unrecogni|stolen|hacked|scam"),
    ("find_overspending", r"overspend|over budget|too much|cut back|save money|"
                          r"spending too|reduce|trim|waste"),
    ("compare_months", r"compare|versus|\bvs\b|last month|previous month|"
                       r"month over month|changed|difference"),
    ("spending_summary", r"spend|spent|breakdown|category|categories|total|"
                         r"subscription|bills|summary|where.*money"),
]


def _extract_months(question: str, months: list[str]) -> list[str]:
    """
    Find every month named in the question, oldest first.

    Returning a list rather than a single month matters. An earlier version
    stopped at the first hit, so "compare March and April" silently dropped
    April, compared February against March instead, and then told the user the
    data did not cover April -- which it did.

    Month names are matched on word boundaries because "may" is a common English
    word; "how much may I spend" should not select 2026-05.
    """
    explicit = [m for m in re.findall(r"20\d{2}-(?:0[1-9]|1[0-2])", question)
                if m in months]
    if explicit:
        return sorted(set(explicit))

    lowered = question.lower()
    hits: list[tuple[int, str]] = []
    for name, num in MONTH_NAMES.items():
        found = re.search(rf"\b{name}\b", lowered)
        if not found:
            continue
        if name == "may" and _may_is_a_verb(lowered, found.end()):
            continue
        matching = [m for m in months if int(m.split("-")[1]) == num]
        if matching:
            hits.append((found.start(), matching[-1]))

    return sorted({m for _, m in hits})


# "May" is the one month that is also an everyday English word, and treating
# "how much may I spend" as a request about 2026-05 produces a confidently wrong
# answer. If the next word is one of these, it is the verb, not the month.
_MAY_VERB_FOLLOWERS = {"i", "we", "you", "he", "she", "it", "they", "there",
                       "be", "have", "not", "need", "want", "get", "still",
                       "also", "well", "the", "my"}


def _may_is_a_verb(lowered: str, after: int) -> bool:
    rest = lowered[after:].strip().split()
    return bool(rest) and rest[0].strip(".,?!") in _MAY_VERB_FOLLOWERS


def route_with_rules(question: str, months: list[str]) -> dict:
    """
    Keyword routing. No LLM, no network, no failure mode.

    This is not a toy backup. It answers most real questions correctly, and it
    is what runs when quota is gone five minutes before a presentation.
    """
    lowered = question.lower()
    tool, matched = "spending_summary", False
    for candidate, pattern in RULE_PATTERNS:
        if re.search(pattern, lowered):
            tool, matched = candidate, True
            break

    args: dict = {}
    found = _extract_months(question, months)
    relative = "last month" in lowered or "previous month" in lowered

    if tool == "compare_months":
        if len(found) >= 2:
            # "March and April" -> compare the later against the earlier.
            args["previous"], args["current"] = found[0], found[-1]
        elif len(found) == 1:
            args["current"] = found[0]
        # "this month vs last month" needs no arguments at all: the defaults
        # already mean latest against the one before. Feeding "last month" in as
        # `current` shifted the whole comparison back a month.
    else:
        if found:
            args["month"] = found[-1]
        elif relative:
            args["month"] = months[-2] if len(months) > 1 else months[-1]

    # `matched` distinguishes a real keyword hit from falling back to the
    # default. Only a real hit is trustworthy enough to skip the model.
    return {"tool": tool, "args": args, "matched": matched,
            "reason": "keyword match (no LLM)", "router": "rules"}


def route_with_llm(question: str, months: list[str]) -> dict:
    """Ask the model which tool to use. Falls back to rules on any problem."""
    # A compact menu rather than pretty-printed JSON. The indented version ran
    # to roughly forty lines and every one of them was tokens the model had to
    # read before it could answer, on the call that sits between the user
    # pressing enter and anything at all appearing on screen.
    menu = "\n".join(
        f"- {s['name']}({', '.join(s['args'])}): {s['use_when']}"
        for s in TOOL_SPECS)

    prompt = (f"Months {months[0]} to {months[-1]}, latest is {months[-1]}.\n"
              f"Tools:\n{menu}\n"
              f"Question: {question}")

    try:
        raw = complete(prompt, system_instruction=ROUTER_SYSTEM,
                       temperature=0.0, json_mode=True, max_output_tokens=400)
        choice = json.loads(raw)
    except Exception as exc:                                # noqa: BLE001
        # Any failure at all -- no key, quota gone, truncated JSON, network
        # blip -- degrades to keyword routing rather than raising. The reason is
        # recorded so a silent fallback is still a visible one when debugging.
        fallback = route_with_rules(question, months)
        fallback["reason"] += f" (llm router failed: {type(exc).__name__})"
        return fallback

    # Never trust the model's output shape. A hallucinated tool name here would
    # be a KeyError in production, so validate before we act on it.
    if choice.get("tool") not in TOOLS:
        return route_with_rules(question, months)

    allowed = {s["name"]: set(s["args"]) for s in TOOL_SPECS}[choice["tool"]]
    args = {k: v for k, v in (choice.get("args") or {}).items()
            if k in allowed and v is not None}
    for key in ("month", "current", "previous"):
        if key in args and args[key] not in months:
            args.pop(key)                      # invented month -> use default

    return {"tool": choice["tool"], "args": args,
            "reason": choice.get("reason", ""), "router": "llm"}


# ==========================================================================
# Answering
# ==========================================================================

ANSWER_SYSTEM = """You are a personal finance assistant.

You will receive a QUESTION and a DATA block computed from the user's real
transactions by pandas.

Rules:
- Every number you write must appear VERBATIM in DATA.
- Do NOT do arithmetic. Never add, subtract, average, total or convert numbers
  to produce a figure that is not already written in DATA. If the total you want
  is not there, quote the totals that are and say what they cover.
- If DATA does not answer the question, say so plainly in one sentence.
- Lead with the direct answer, then at most two supporting details.
- Keep it under 90 words. Write plainly, no bullet lists, no preamble.
- Money as EGP 1,234.56. Do not repeat the whole table back."""


def _template_answer(question: str, result: dict) -> str:
    """Fallback prose when no LLM is available. Plain, but never wrong."""
    return f"{result['headline']}\n\n{result['data_text']}"


def plan(df: pd.DataFrame, question: str, use_llm: bool = True) -> tuple[dict, dict]:
    """
    Route the question and run the chosen tool -- everything except the writing.

    Split out from ask() so the interface can show the computed table while the
    prose is still streaming in.
    """
    months = analytics.available_months(df)

    # Fast path. Routing was measured at ~1.5s, roughly half the total wait, and
    # for a question containing an unambiguous word like "overspending",
    # "suspicious" or "compare" a regex reaches the same answer in microseconds.
    # Spending a second and a half asking a model to confirm what a keyword
    # already settled is waste the user pays for on every question.
    #
    # The model is still what handles "im bleeding money somewhere" or a
    # question in another language -- anything the keywords do not positively
    # match falls through to it. Fast when it can be, smart when it must be.
    fast = route_with_rules(question, months)
    if fast["matched"]:
        choice = fast
        choice["router"] = "rules (fast path)"
    elif use_llm and is_available():
        choice = route_with_llm(question, months)
    else:
        choice = fast

    try:
        result = TOOLS[choice["tool"]](df, **choice["args"])
    except Exception as exc:                                # noqa: BLE001
        # A bad argument should not kill the answer; retry with defaults.
        result = TOOLS[choice["tool"]](df)
        choice["reason"] += f" (args rejected: {type(exc).__name__})"

    return choice, result


def answer_stream(question: str, result: dict):
    """Yield the written answer piece by piece, for a live-typing effect."""
    yield from complete_stream(
        f"QUESTION: {question}\n\nDATA:\n{result['data_text']}",
        system_instruction=ANSWER_SYSTEM, temperature=0.2)


def ask(df: pd.DataFrame, question: str, use_llm: bool = True) -> dict:
    """
    Answer one question in one go. Returns the prose plus everything behind it.

    The extra keys are not decoration -- the Streamlit UI shows which tool ran
    and renders the table, so a user can check the agent's working rather than
    taking its word for it.
    """
    choice, result = plan(df, question, use_llm)
    answer, source = _template_answer(question, result), "template"
    if use_llm and is_available():
        try:
            answer = complete(
                f"QUESTION: {question}\n\nDATA:\n{result['data_text']}",
                system_instruction=ANSWER_SYSTEM, temperature=0.2)
            source = "llm"
        except LLMUnavailable:
            pass

    return {"question": question, "answer": answer, "tool": choice["tool"],
            "args": choice["args"], "router": choice.get("router", "rules"),
            "answer_source": source, "headline": result["headline"],
            "table": result["table"], "data_text": result["data_text"]}


# ==========================================================================
# Demo
# ==========================================================================

DEMO_QUESTIONS = [
    "Where am I overspending?",
    "How does this month compare to last month?",
    "Are there any suspicious transactions on my account?",
    "What did I spend the most on in July?",
    "How much am I paying for subscriptions?",
    "What will Bitcoin be worth next year?",
]


def main() -> None:
    df = load_transactions()

    print(f"LLM available: {is_available()}\n")
    for question in DEMO_QUESTIONS:
        out = ask(df, question)
        print("=" * 74)
        print(f"Q: {question}")
        print(f"   [tool: {out['tool']} | router: {out['router']} | "
              f"answer: {out['answer_source']}]")
        print("=" * 74)
        print(out["answer"])
        print()


if __name__ == "__main__":
    main()
