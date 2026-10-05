"""
Spending analytics -- the pandas layer underneath the chatbot.

    python -m src.analytics        # prints a demo of every function

Design rule: every function here returns *data* (a DataFrame or a dict), never a
formatted sentence. The language model phrases things; pandas decides what is
true. Keeping that boundary sharp is what stops the agent from inventing a
number, because it never computes one -- it only ever reads one it was handed.

These functions are also the agent's toolbox. Their signatures matter more than
their internals: an LLM has to be able to look at a name and a couple of
arguments and know when to reach for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import load_transactions


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _drop_movements(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Remove money that never left the person's control.

    Transfers between your own accounts, cash withdrawals and credit-card
    payments all appear as debits, but none of them is spending -- the money has
    only moved. Counting them double-counts: the transfer *and* whatever the
    cash was later spent on. Exports that mark these rows are the reason this
    exists; when nothing is marked, nothing is dropped.
    """
    if "is_movement" in frame.columns:
        return frame[frame["is_movement"] == 0]
    return frame


def spending(df: pd.DataFrame) -> pd.DataFrame:
    """Money out, excluding movements between the user's own accounts."""
    return _drop_movements(df[df["direction"] == "debit"])


def income(df: pd.DataFrame) -> pd.DataFrame:
    """Money in, excluding movements between the user's own accounts."""
    return _drop_movements(df[df["direction"] == "credit"])


def available_months(df: pd.DataFrame) -> list[str]:
    return sorted(df["month"].unique())


def latest_month(df: pd.DataFrame) -> str:
    return available_months(df)[-1]


def _previous_month(df: pd.DataFrame, month: str) -> str | None:
    months = available_months(df)
    i = months.index(month)
    return months[i - 1] if i > 0 else None


# --------------------------------------------------------------------------
# Core analytics
# --------------------------------------------------------------------------

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Income, spending, net and savings rate for every month.

    The savings rate is the number a person actually feels: what fraction of
    what came in did not go straight back out. Negative means the month was
    funded from savings.
    """
    months = sorted(df["month"].unique())
    out = pd.DataFrame(index=pd.Index(months, name="month"))
    out["income"] = (income(df).groupby("month")["amount"].sum()
                     .reindex(months, fill_value=0.0))
    out["spending"] = (spending(df).groupby("month")["amount"].sum()
                       .reindex(months, fill_value=0.0))
    out["net"] = out["income"] - out["spending"]
    out["savings_rate"] = np.where(out["income"] > 0,
                                   out["net"] / out["income"] * 100, np.nan)
    return out.round(2)


def category_breakdown(df: pd.DataFrame, month: str | None = None) -> pd.DataFrame:
    """Where the money went, for one month or for the whole period."""
    s = spending(df)
    if month:
        s = s[s["month"] == month]
    if s.empty:
        return pd.DataFrame(columns=["total", "transactions", "average", "share_pct"])

    out = s.groupby("category")["amount"].agg(
        total="sum", transactions="count", average="mean")
    out["share_pct"] = out["total"] / out["total"].sum() * 100
    return out.sort_values("total", ascending=False).round(2)


def spending_trend(df: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    """Monthly spending totals, optionally for a single category."""
    s = spending(df)
    if category:
        s = s[s["category"] == category]
    out = s.groupby("month")["amount"].agg(total="sum", transactions="count")
    out["change_pct"] = out["total"].pct_change() * 100
    return out.round(2)


def compare_months(df: pd.DataFrame, current: str,
                   previous: str | None = None) -> pd.DataFrame:
    """
    Category-by-category difference between two months.

    Sorted by absolute dollar change rather than percentage, deliberately. A
    category going from $4 to $12 is +200% and irrelevant; one going from $520
    to $760 is +46% and is the thing worth mentioning.
    """
    previous = previous or _previous_month(df, current)
    if previous is None:
        raise ValueError(f"No month available before {current}")

    cur = category_breakdown(df, current)["total"]
    prev = category_breakdown(df, previous)["total"]

    out = pd.DataFrame({previous: prev, current: cur}).fillna(0.0)
    out["change"] = out[current] - out[previous]
    out["change_pct"] = np.where(
        out[previous] > 0, out["change"] / out[previous] * 100, np.nan)
    return out.reindex(out["change"].abs().sort_values(ascending=False).index).round(2)


def overspending_signals(df: pd.DataFrame, month: str | None = None,
                         lookback: int = 3, min_change: float = 25.0
                         ) -> pd.DataFrame:
    """
    The direct answer to "where am I overspending?".

    Each category in the chosen month is compared against its own average over
    the preceding `lookback` months, so the baseline is this person's habits
    rather than some external budget they never agreed to. Categories are ranked
    by dollar overshoot; anything moving less than `min_change` is dropped as
    noise not worth a sentence.
    """
    month = month or latest_month(df)
    months = available_months(df)
    i = months.index(month)
    baseline_months = months[max(0, i - lookback):i]
    if not baseline_months:
        raise ValueError(f"No history before {month} to compare against")

    current = category_breakdown(df, month)["total"]

    hist = spending(df)
    hist = hist[hist["month"].isin(baseline_months)]
    baseline = (hist.groupby(["month", "category"])["amount"].sum()
                    .unstack(fill_value=0.0)
                    .reindex(columns=current.index, fill_value=0.0)
                    .mean())

    out = pd.DataFrame({"this_month": current, "usual": baseline}).fillna(0.0)
    out["over_by"] = out["this_month"] - out["usual"]
    out["over_by_pct"] = np.where(
        out["usual"] > 0, out["over_by"] / out["usual"] * 100, np.nan)

    out = out[out["over_by"].abs() >= min_change]
    return out.sort_values("over_by", ascending=False).round(2)


def top_merchants(df: pd.DataFrame, n: int = 10,
                  month: str | None = None) -> pd.DataFrame:
    """Biggest merchants by total spend."""
    s = spending(df)
    if month:
        s = s[s["month"] == month]
    out = s.groupby("merchant")["amount"].agg(
        total="sum", transactions="count", average="mean")
    return out.sort_values("total", ascending=False).head(n).round(2)


def recurring_charges(df: pd.DataFrame, min_months: int = 5,
                      max_variation: float = 0.25,
                      max_per_month: float = 1.3) -> pd.DataFrame:
    """
    Find subscriptions and standing bills.

    Three conditions, and all three are needed:

    * appears in at least `min_months` distinct months
    * the amount barely moves (coefficient of variation under `max_variation`)
    * bills roughly **once** per month (`max_per_month`)

    That third test is the one that is easy to forget and it matters. Without
    it this function confidently reported Shell Oil as a subscription: 57 fuel
    stops across 12 months, present every month, and consistently around $43
    because a tank of petrol costs what a tank of petrol costs. Frequency is
    what separates a standing charge from a regular habit.

    "You are committed to $X every month before you decide anything" is often
    the single most actionable line a finance tool can produce, so it is worth
    the extra condition to get it right.
    """
    s = spending(df)
    grouped = s.groupby("merchant").agg(
        months=("month", "nunique"),
        charges=("amount", "size"),
        mean_amount=("amount", "mean"),
        std_amount=("amount", "std"),
    )
    grouped["std_amount"] = grouped["std_amount"].fillna(0.0)
    grouped["variation"] = grouped["std_amount"] / grouped["mean_amount"]
    grouped["per_month"] = grouped["charges"] / grouped["months"]

    out = grouped[(grouped["months"] >= min_months)
                  & (grouped["variation"] <= max_variation)
                  & (grouped["per_month"] <= max_per_month)].copy()
    out["monthly_cost"] = out["mean_amount"]
    return out[["months", "charges", "monthly_cost", "variation"]] \
        .sort_values("monthly_cost", ascending=False).round(3)


def budget_health(df: pd.DataFrame) -> dict:
    """One-glance overview of the whole period."""
    monthly = monthly_summary(df)
    s = spending(df)
    cats = category_breakdown(df)

    return {
        "period": f"{df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}",
        "transactions": int(len(df)),
        "total_income": float(income(df)["amount"].sum().round(2)),
        "total_spending": float(s["amount"].sum().round(2)),
        "net": float((income(df)["amount"].sum() - s["amount"].sum()).round(2)),
        "avg_monthly_income": float(monthly["income"].mean().round(2)),
        "avg_monthly_spending": float(monthly["spending"].mean().round(2)),
        "avg_savings_rate_pct": float(monthly["savings_rate"].mean().round(1)),
        "months_in_deficit": int((monthly["net"] < 0).sum()),
        "months_total": int(len(monthly)),
        # An account can legitimately have no spending rows -- a statement of
        # nothing but transfers, or a file the app misread. Asking for the
        # biggest category of nothing used to raise IndexError and take the
        # whole page down before the user saw a single number.
        "largest_category": str(cats.index[0]) if not cats.empty else "none",
        "largest_category_total": (float(cats.iloc[0]["total"])
                                   if not cats.empty else 0.0),
    }


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

def main() -> None:
    df = load_transactions()
    month = latest_month(df)

    print("=" * 74)
    print("BUDGET HEALTH")
    print("=" * 74)
    for key, value in budget_health(df).items():
        label = key.replace("_", " ").capitalize()
        formatted = f"{value:,.2f}" if isinstance(value, float) else value
        print(f"  {label:26s} {formatted}")

    print("\n" + "=" * 74)
    print("MONTHLY SUMMARY")
    print("=" * 74)
    print(monthly_summary(df).to_string())

    print("\n" + "=" * 74)
    print(f"CATEGORY BREAKDOWN -- {month}")
    print("=" * 74)
    print(category_breakdown(df, month).to_string())

    print("\n" + "=" * 74)
    print(f"WHERE AM I OVERSPENDING? -- {month} vs previous 3 months")
    print("=" * 74)
    print(overspending_signals(df, month).to_string())

    print("\n" + "=" * 74)
    print(f"MONTH-OVER-MONTH -- {month}")
    print("=" * 74)
    print(compare_months(df, month).to_string())

    print("\n" + "=" * 74)
    print("RECURRING CHARGES / SUBSCRIPTIONS")
    print("=" * 74)
    rec = recurring_charges(df)
    print(rec.to_string())
    print(f"\n  Total committed each month: EGP {rec['monthly_cost'].sum():,.2f}")

    print("\n" + "=" * 74)
    print("TOP MERCHANTS (whole period)")
    print("=" * 74)
    print(top_merchants(df, 8).to_string())


if __name__ == "__main__":
    main()
