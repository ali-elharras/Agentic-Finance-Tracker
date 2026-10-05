"""
Shared data loading and feature engineering.

Everything downstream -- the categoriser (Sprint 2), the fraud detector
(Sprint 3), the analytics engine (Sprint 4) -- loads data through here, so all
of them see identical columns. When feature logic lives in three places it
drifts, and a model trained on one version silently breaks on another.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

DATA_PATH = "data/processed/transactions.csv"

# Columns the categoriser is allowed to see.
#
# `merchant` is deliberately absent. It holds the clean name ("Walmart
# Supercenter"), but a real bank feed only supplies the raw descriptor
# ("POS DEBIT WALMART SUPERCE #6283"). Training on the clean name would score
# far higher and be useless in production -- the model would never learn to
# read a descriptor, which is the actual job.
TEXT_COL = "description"
NUMERIC_COLS = ["amount", "log_amount", "hour", "day_of_week",
                "is_weekend", "day_of_month"]
CATEGORICAL_COLS = ["channel"]


def load_transactions(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return add_time_features(df)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break the timestamp into parts a model can use.

    A raw timestamp is nearly useless to a classifier -- it is one enormous
    ever-increasing number. Split into hour / weekday / day-of-month it becomes
    genuinely predictive: coffee happens at 8am, rent on the 1st, restaurants at
    weekends.
    """
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # Monday = 0
    df["day_of_month"] = ts.dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = ts.dt.to_period("M").astype(str)

    # Amounts are heavily right-skewed: hundreds of $5 coffees, a few $500
    # flights. The log compresses that range so the big values stop dominating
    # distance-based calculations.
    df["log_amount"] = np.log1p(df["amount"])
    return df


_DESC_PREFIXES = ["FAWRY BILL PAY", "VISA PURCHASE", "INSTAPAY TRF",
                  "SALARY CREDIT", "AUTO DEBIT", "MEEZA POS", "WEB PMT",
                  "ACH DEBIT", "POS DEBIT", "POS"]

# Cairo districts that appear at the end of card descriptors and are location,
# not identity: "VISA PURCHASE CILANTRO MAADI" and "... CILANTRO ZAMALEK" are
# the same coffee chain.
_DESC_AREAS = ["SHEIKH ZAYED", "6TH OCTOBER", "NASR CITY", "NEW CAIRO",
               "HELIOPOLIS", "MOHANDESEEN", "ZAMALEK", "MOKATTAM", "MANIAL",
               "MAADI", "DOKKI", "GIZA"]


def normalise_merchant(description: str) -> str:
    """
    Recover a merchant key from a raw bank descriptor.

    Needed because an uploaded statement has no clean merchant column, and
    grouping by the raw descriptor is useless: every bill carries a fresh
    reference number, so "FAWRY BILL PAY VODAFONE EGYPT 481920355" never
    matches next month's. Without this, subscription detection found nothing
    real and instead reported "VISA PURCHASE ZOOBA GIZA" as a monthly
    commitment, because that string happened to repeat.

    Strips the scheme prefix, reference digits, ".COM", the country tag and the
    district, then caps the result at 14 characters so the bank's random
    truncation ("NORTH CAIRO ELECTRICITY" vs "NORTH CAIRO ELECT") collapses to
    one key.
    """
    text = str(description).upper()
    for prefix in _DESC_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.replace(".COM", " ")
    text = re.sub(r"\b\d{3,}\b", " ", text)          # reference numbers
    for area in _DESC_AREAS:
        text = text.replace(area, " ")
    text = re.sub(r"\b(EG|EGYPT)\b", " ", text)
    text = re.sub(r"[^A-Z0-9& ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:14].strip() or str(description).upper()[:14])


def spending_only(df: pd.DataFrame) -> pd.DataFrame:
    """Money going out. Salary is income, not a spending category."""
    return df[df["direction"] == "debit"].reset_index(drop=True)


def time_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Split chronologically: oldest rows train, newest rows test.

    Not a random split. In production the model always predicts the future from
    the past, so the test set has to be the future. A random split scatters
    August and July rows into both halves, the model quietly learns
    period-specific patterns it would never have at prediction time, and the
    reported score comes out optimistic.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def label_ceiling(df: pd.DataFrame, label: str = "category",
                  min_rows: int = 3) -> float:
    """
    A rough estimate of the best accuracy any model could reach on this data.

    Two things cap it. Merchants are ambiguous -- Carrefour is ~72% Groceries
    and ~28% Shopping, so even perfect knowledge of "this is Carrefour" leaves
    you guessing on a quarter of them. And 3% of labels are wrong on purpose.

    Read this as a guide, not a hard limit. For merchants appearing fewer than
    `min_rows` times it assumes the best you can do is the overall prior, which
    is deliberately pessimistic: a one-off shop called "EL SHAMS PHARMACY"
    announces its own category in its name, and a text model reads that happily.
    So a few points above this estimate is normal and healthy. A model landing
    far above it -- near 100% when 3% of labels are known to be wrong -- is the
    signal that something has leaked.
    """
    prior = df[label].value_counts(normalize=True).iloc[0]
    correct = 0.0
    for _, grp in df.groupby("merchant"):
        if len(grp) >= min_rows:
            # Enough history to know this merchant's most likely category.
            correct += grp[label].value_counts().iloc[0]
        else:
            # Too rare to learn from repetition alone; assume only the prior.
            correct += len(grp) * prior
    return correct / len(df)
