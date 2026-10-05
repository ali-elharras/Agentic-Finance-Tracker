"""
Synthetic bank transaction generator.

Produces ~2,000 transactions spanning 12 months for a single fictional account
holder living in Austin, TX. The output is written to
`data/processed/transactions.csv`.

Run it from the project root:

    python -m src.generate_data

Design notes
------------
Three things make this dataset useful rather than merely fake:

1. *Rhythm.* Rent lands on the 1st, salary twice a month, coffee on weekday
   mornings, restaurants at weekends. Without rhythm the monthly-trend
   analytics in Sprint 4 would have nothing to show.

2. *Ambiguity.* Many merchants map to more than one category (see
   `merchants.py`), roughly 3% of labels are deliberately wrong, and some
   merchants are one-off names the model has never seen. Together these give
   the classifier a genuine problem to solve instead of a lookup table to
   memorise.

3. *Structured fraud.* Fraud arrives as episodes with recognisable shapes --
   card testing, geographic impossibility, late-night spikes, rapid repeats --
   not as randomly scattered expensive rows.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.merchants import (
    FRAUD_MERCHANT_POOL,
    MERCHANT_BY_NAME,
    MERCHANTS,
    RECURRING,
    SPENDING_CATEGORIES,
    draw_amount,
    make_descriptor,
    pick_category,
    pick_channel,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 42                       # fixes randomness so every run is identical
START = datetime(2025, 8, 1)
# End of day, not midnight: a paycheck timestamped 06:00 on the 31st must fall
# inside the window. An earlier version used midnight here and silently dropped
# the final month's salary.
END = datetime(2026, 7, 31, 23, 59, 59)
N_DISCRETIONARY = 1100          # everyday spending, on top of recurring bills
N_FRAUD_EPISODES = 10           # each episode is 1-6 transactions, in its own month
LABEL_NOISE_RATE = 0.03         # share of categories deliberately mislabelled
LONG_TAIL_RATE = 0.08           # share of purchases at never-seen-before merchants

# The persona: a mid-level software engineer in Cairo taking home about
# EGP 45,000 a month, renting a flat in Maadi. Figures are chosen so the year
# roughly balances -- spending a little under income, with some months in
# deficit. A tracker built on someone who outspends their salary 4:1 cannot give
# meaningful budget advice.
HOME_CITY, HOME_COUNTRY = "Cairo", "EG"
MONTHLY_RENT = 12000.00
SEMI_MONTHLY_PAY = 27500.00     # EGP 55,000 a month net

OUT_PATH = "data/processed/transactions.csv"

fake = Faker("en_US")

# Relative likelihood of spending on each weekday (Mon .. Sun).
DOW_WEIGHTS = {
    "Groceries":         [0.9, 0.8, 0.9, 1.0, 1.3, 1.7, 1.4],
    "Food":            [0.9, 0.9, 1.0, 1.1, 1.6, 1.8, 1.3],
    "Transport":         [1.4, 1.4, 1.4, 1.4, 1.3, 0.7, 0.5],
    "Shopping":          [0.8, 0.8, 0.9, 1.0, 1.2, 1.6, 1.5],
    "Entertainment":     [0.6, 0.6, 0.7, 0.9, 1.7, 2.0, 1.3],
    "Health & Fitness":  [1.2, 1.1, 1.2, 1.1, 1.0, 0.9, 0.7],
    "Travel":            [0.8, 0.7, 0.8, 1.0, 1.6, 1.5, 1.2],
    "Bills & Utilities": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

# When during the day each category tends to happen: (start_hour, end_hour, weight).
HOUR_PROFILE = {
    "Groceries":         [(10, 13, 0.3), (16, 20, 0.7)],
    "Food":            [(7, 10, 0.40), (11, 14, 0.30), (18, 21, 0.30)],
    "Transport":         [(6, 9, 0.45), (16, 19, 0.40), (20, 23, 0.15)],
    "Shopping":          [(11, 15, 0.4), (16, 21, 0.6)],
    "Entertainment":     [(17, 23, 1.0)],
    "Health & Fitness":  [(7, 10, 0.35), (11, 18, 0.65)],
    "Travel":            [(8, 20, 1.0)],
    "Bills & Utilities": [(8, 20, 1.0)],
}

# Typical EGP spend for merchants the model has never seen, keyed by category.
# Same tight-tail rule as merchants.py: mean is (low + mode + high) / 3.
CATEGORY_AMOUNT = {
    "Groceries":         (120, 400, 950),
    "Food":            (70, 175, 400),
    "Transport":         (50, 140, 360),
    "Shopping":          (200, 620, 1700),
    "Bills & Utilities": (200, 500, 1050),
    "Entertainment":     (110, 270, 660),
    "Health & Fitness":  (120, 370, 950),
    "Travel":            (1200, 2900, 6500),
}

# Small independent shops, the sort that appear once or twice on a statement.
# Faker cannot help here -- its ar_EG locale falls back to English company names
# like "Carroll Inc", which would be an obvious tell in an Egyptian dataset -- so
# these are composed from Egyptian naming conventions instead. The shop type also
# fixes the category, which is both realistic and learnable: a classifier can
# work out that PHARMACY means Health without ever seeing that shop before.
LT_PREFIX = ["EL", "AL", "", ""]
LT_CORE = ["SHAMS", "NASR", "AMEER", "HORREYA", "SALAM", "MAHROUSA", "YASMEEN",
           "ZAHRA", "FIRDAUS", "KARNAK", "NILE", "RAMSES", "TAHRIR", "GEZIRA",
           "MOKATTAM", "ANDALUS", "SAFA", "MARWA", "REHAB", "ORMAN"]
LT_TYPE = {
    "MARKET": "Groceries", "SUPERMARKET": "Groceries", "GROCERY": "Groceries",
    "PHARMACY": "Health & Fitness", "LAB": "Health & Fitness",
    "CAFE": "Food", "RESTAURANT": "Food", "BAKERY": "Food",
    "STORES": "Shopping", "BOUTIQUE": "Shopping", "CENTER": "Shopping",
}

# Where a stolen Egyptian card gets used. Real cardholders rarely shop abroad
# overnight, and almost never in these places on the same day they bought coffee
# in Maadi.
FOREIGN_LOCATIONS = [
    ("Istanbul", "TR"), ("Lagos", "NG"), ("Jakarta", "ID"),
    ("Kyiv", "UA"), ("Manila", "PH"), ("Bucharest", "RO"),
    ("Sao Paulo", "BR"), ("Ho Chi Minh City", "VN"),
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def pick_hour(category: str) -> int:
    """Choose an hour of day that fits how people actually spend."""
    segments = HOUR_PROFILE.get(category, [(9, 20, 1.0)])
    lo, hi, _ = random.choices(segments, weights=[s[2] for s in segments], k=1)[0]
    return random.randint(lo, hi)


def random_datetime(category: str) -> datetime:
    """Pick a date weighted by weekday, then a plausible time on that date."""
    span_days = (END - START).days
    weights = DOW_WEIGHTS.get(category, [1.0] * 7)
    for _ in range(20):
        day = START + timedelta(days=random.randint(0, span_days))
        if random.random() < weights[day.weekday()] / 2.0:
            break
    return day.replace(hour=pick_hour(category),
                       minute=random.randint(0, 59),
                       second=random.randint(0, 59))


def make_row(*, ts, description, merchant, amount, direction, category,
             channel, city, country, is_fraud) -> dict:
    return {
        "timestamp": ts,
        "description": description,
        "merchant": merchant,
        "amount": round(float(amount), 2),
        "direction": direction,
        "category": category,
        "channel": channel,
        "city": city,
        "country": country,
        "is_fraud": is_fraud,
    }


def month_starts(start: datetime, end: datetime):
    cur = datetime(start.year, start.month, 1)
    while cur <= end:
        yield cur
        cur = datetime(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)


# --------------------------------------------------------------------------
# 1. Recurring money: salary, rent, subscriptions, bills
# --------------------------------------------------------------------------

def generate_recurring() -> list[dict]:
    rows: list[dict] = []

    for m0 in month_starts(START, END):
        # --- Salary, paid on the 15th and the last day of the month ---
        last_day = (datetime(m0.year + (m0.month == 12), (m0.month % 12) + 1, 1)
                    - timedelta(days=1))
        for pay_day in (m0.replace(day=15), last_day):
            if not (START <= pay_day <= END):
                continue
            rows.append(make_row(
                ts=pay_day.replace(hour=6, minute=random.randint(0, 30)),
                description=f"SALARY CREDIT NILEWARE TECH EGYPT "
                            f"{fake.numerify('########')}",
                merchant="Nileware Tech Egypt Payroll",
                amount=round(SEMI_MONTHLY_PAY + random.uniform(-45, 45), 2),
                direction="credit", category="Income", channel="transfer",
                city=HOME_CITY, country=HOME_COUNTRY, is_fraud=0,
            ))

        # --- Rent, first of the month ---
        rent_day = m0.replace(hour=9, minute=random.randint(0, 59))
        if START <= rent_day <= END:
            rows.append(make_row(
                ts=rent_day,
                description=f"INSTAPAY TRF RENT MAADI "
                            f"{fake.numerify('##########')}",
                merchant="Flat Rent Maadi",
                amount=MONTHLY_RENT, direction="debit",
                category="Bills & Utilities", channel="transfer",
                city=HOME_CITY, country=HOME_COUNTRY, is_fraud=0,
            ))

        # --- Subscriptions and utility bills ---
        for bill in RECURRING:
            merchant = MERCHANT_BY_NAME[bill["name"]]
            day_num = min(bill["day"] + random.randint(-bill["jitter"], bill["jitter"]), 28)
            ts = m0.replace(day=max(1, day_num), hour=random.randint(1, 6),
                            minute=random.randint(0, 59))
            if not (START <= ts <= END):
                continue
            rows.append(make_row(
                ts=ts,
                description=make_descriptor(merchant["name"], merchant["style"]),
                merchant=merchant["name"],
                amount=draw_amount(merchant), direction="debit",
                category=pick_category(merchant), channel="online",
                city=HOME_CITY, country=HOME_COUNTRY, is_fraud=0,
            ))

    return rows


# --------------------------------------------------------------------------
# 2. Everyday discretionary spending
# --------------------------------------------------------------------------

def generate_discretionary(n: int) -> list[dict]:
    rows: list[dict] = []
    pool = [m for m in MERCHANTS if m["weight"] > 0]
    weights = [m["weight"] for m in pool]

    for _ in range(n):
        if random.random() < LONG_TAIL_RATE:
            # A small shop the model has never seen. It has to work the category
            # out from the shop type in the text plus the amount -- exactly the
            # generalisation we want the classifier tested on.
            shop_type = random.choice(list(LT_TYPE))
            category = LT_TYPE[shop_type]
            name = " ".join(x for x in (random.choice(LT_PREFIX),
                                        random.choice(LT_CORE),
                                        shop_type) if x)
            style = random.choice(["pos", "visa", "meeza"])
            lo, mode, hi = CATEGORY_AMOUNT[category]
            amount = round(random.triangular(lo, hi, mode), 2)
            channel = "card_present"
        else:
            merchant = random.choices(pool, weights=weights, k=1)[0]
            category = pick_category(merchant)
            name = merchant["name"]
            style = merchant["style"]
            amount = draw_amount(merchant)
            channel = pick_channel(merchant)

        rows.append(make_row(
            ts=random_datetime(category),
            description=make_descriptor(name, style),
            merchant=name, amount=amount, direction="debit",
            category=category, channel=channel,
            city=HOME_CITY, country=HOME_COUNTRY, is_fraud=0,
        ))

    return rows


# --------------------------------------------------------------------------
# 3. Fraud episodes
# --------------------------------------------------------------------------

def _fraud_merchant():
    name, category, (lo, hi) = random.choice(FRAUD_MERCHANT_POOL)
    return name, category, round(random.uniform(lo, hi), 2)


def _episode_start(month: datetime) -> datetime:
    """
    A random day inside one specific month.

    Episodes are pinned to distinct months on purpose. Drawing days uniformly
    across the whole year let several episodes collide in the same month and
    doubled that month's spending, which would have shown up in Sprint 4 as a
    spending trend rather than as fraud.
    """
    return month.replace(day=random.randint(2, 26),
                         hour=random.randint(0, 23),
                         minute=random.randint(0, 59))


def generate_fraud(n_episodes: int) -> list[dict]:
    """
    Four fraud shapes, each mirroring a real-world attack pattern.

    The point is that no single column gives fraud away. Card testing is cheap,
    not expensive. A foreign purchase is normal for a traveller. 3 a.m. activity
    is odd but not proof. Only the *combination* is suspicious -- which is why
    Sprint 3 needs a model rather than an if-statement.
    """
    rows: list[dict] = []
    shapes = ["card_testing", "foreign_burst", "late_night_spike", "rapid_repeat"]

    # One episode per month at most, in a random subset of months, so a few
    # months stay completely clean. Those clean months are what give Sprint 3
    # something to contrast against.
    months = list(month_starts(START, END))
    chosen = sorted(random.sample(range(len(months)), k=min(n_episodes, len(months))))

    for i, month_idx in enumerate(chosen):
        shape = shapes[i % len(shapes)]
        t0 = _episode_start(months[month_idx])

        if shape == "card_testing":
            # Small probing charges to check the card is live, then the real hit.
            for k in range(random.randint(3, 5)):
                ts = t0 + timedelta(minutes=k * random.randint(2, 7))
                rows.append(make_row(
                    ts=ts.replace(hour=random.randint(0, 5)),
                    description=f"WEB PMT DIGITALGOODS EG "
                                f"{fake.numerify('#####')}",
                    merchant="Digital Goods Online",
                    # Card testing: tiny probe charges to check the card is live.
                    amount=round(random.uniform(5, 45), 2), direction="debit",
                    category="Shopping", channel="online",
                    city=HOME_CITY, country=HOME_COUNTRY, is_fraud=1,
                ))
            name, category, amount = _fraud_merchant()
            rows.append(make_row(
                ts=(t0 + timedelta(minutes=40)).replace(hour=random.randint(0, 5)),
                description=make_descriptor(name, "online"),
                merchant=name, amount=amount, direction="debit",
                category=category, channel="online",
                city=HOME_CITY, country=HOME_COUNTRY, is_fraud=1,
            ))

        elif shape == "foreign_burst":
            city, country = random.choice(FOREIGN_LOCATIONS)
            for k in range(random.randint(2, 4)):
                name, category, amount = _fraud_merchant()
                ts = t0 + timedelta(hours=k, minutes=random.randint(0, 50))
                rows.append(make_row(
                    ts=ts.replace(hour=random.randint(0, 23)),
                    description=make_descriptor(name, "pos"),
                    merchant=name, amount=amount, direction="debit",
                    category=category, channel="card_present",
                    city=city, country=country, is_fraud=1,
                ))

        elif shape == "late_night_spike":
            for k in range(random.randint(1, 2)):
                name, category, amount = _fraud_merchant()
                ts = (t0 + timedelta(minutes=k * 25)).replace(
                    hour=random.randint(2, 4), minute=random.randint(0, 59))
                rows.append(make_row(
                    ts=ts,
                    description=make_descriptor(name, "online"),
                    merchant=name, amount=round(amount * random.uniform(1.2, 2.0), 2),
                    direction="debit", category=category, channel="online",
                    city=HOME_CITY, country=HOME_COUNTRY, is_fraud=1,
                ))

        else:  # rapid_repeat -- same merchant charged several times in minutes
            name, category, amount = _fraud_merchant()
            for k in range(3):
                ts = t0 + timedelta(minutes=k * random.randint(1, 4))
                rows.append(make_row(
                    ts=ts.replace(hour=random.randint(0, 23)),
                    description=make_descriptor(name, "online"),
                    merchant=name, amount=amount, direction="debit",
                    category=category, channel="online",
                    city=HOME_CITY, country=HOME_COUNTRY, is_fraud=1,
                ))

    return rows


# --------------------------------------------------------------------------
# 4. Label noise
# --------------------------------------------------------------------------

def add_label_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip a small share of category labels to something wrong.

    Real users miscategorise their own spending constantly. Modelling that has a
    useful side effect: it puts a ceiling on achievable accuracy. With 3% of
    labels wrong, a perfect model tops out near 97% -- so if Sprint 2 reports
    99%, that is evidence of a bug, not of brilliance.
    """
    spend = df.index[(df["direction"] == "debit") & (df["is_fraud"] == 0)]
    n_flip = int(len(spend) * LABEL_NOISE_RATE)
    for idx in np.random.choice(spend, size=n_flip, replace=False):
        current = df.at[idx, "category"]
        options = [c for c in SPENDING_CATEGORIES if c != current]
        df.at[idx, "category"] = random.choice(options)
    return df


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------

def main() -> pd.DataFrame:
    random.seed(SEED)
    np.random.seed(SEED)
    Faker.seed(SEED)

    rows = generate_recurring() + generate_discretionary(N_DISCRETIONARY) \
        + generate_fraud(N_FRAUD_EPISODES)

    df = pd.DataFrame(rows)
    df = df[(df["timestamp"] >= START) & (df["timestamp"] <= END)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_label_noise(df)

    df.insert(0, "transaction_id", [f"TXN{i:06d}" for i in range(1, len(df) + 1)])
    df.to_csv(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    frame = main()
    print(f"Wrote {len(frame):,} transactions to {OUT_PATH}")
    print(f"Date range : {frame['timestamp'].min()} -> {frame['timestamp'].max()}")
    print(f"Fraud rows : {int(frame['is_fraud'].sum())} "
          f"({frame['is_fraud'].mean():.2%})")
