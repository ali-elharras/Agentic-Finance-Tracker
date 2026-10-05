"""
Fraud / anomaly detection.

    python -m src.train_fraud_detector

Two detectors are built and compared:

1. **Z-score** -- a purely statistical baseline. Flag anything whose amount is
   more than N standard deviations from normal *for its category*. Simple,
   explainable, and blind to everything except size.

2. **Isolation Forest** -- an unsupervised model that flags rows which are easy
   to separate from the rest of the data across *all* features at once.

The critical detail: neither detector is ever shown the `is_fraud` column.
Isolation Forest is unsupervised -- it hunts for "unusual", not for "fraud", and
has no idea the label exists. We use `is_fraud` only afterwards, as an answer
key, to measure how well "unusual" lined up with "criminal".

That matters for your write-up. A real bank does not have labels for fraud that
has not been reported yet, which is exactly why unsupervised detection earns its
place alongside a supervised classifier.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import load_transactions, spending_only

MODEL_PATH = "models/fraud_detector.pkl"
SEED = 42
CONTAMINATION = 0.03      # our prior guess at the share of anomalies
Z_THRESHOLD = 3.0

FRAUD_FEATURES = [
    "log_amount",
    "hour",
    "is_night",
    "is_weekend",
    "is_foreign",
    "is_online",
    "amount_z_in_category",
    "log_minutes_since_prev",
    "txns_last_hour",
    "amount_vs_personal_median",
]


def add_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the signals a fraud analyst would actually look at.

    Each one targets a real attack shape. None of them is decisive alone --
    that is the whole point. Legitimate people shop online, travel, and
    occasionally buy something expensive at midnight.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Plenty of real exports carry a date but no time. pandas reads those as
    # midnight, which makes every single row look like a 3am transaction and
    # every gap between them zero. Two of the ten signals then fire on all
    # 200 rows and drown out the eight that still mean something, so when there
    # is no clock in the data we switch them off rather than trust them.
    has_time = int(df["timestamp"].dt.time.nunique() > 1)
    df["has_time"] = has_time

    # "Foreign" means away from wherever this account normally transacts, taken
    # as the most common country in the data. Hardcoding a home country made
    # every single Egyptian transaction read as foreign, and the explanations
    # dutifully reported "foreign transaction (Cairo, EG)" on all of them.
    home = (df["country"].mode().iloc[0]
            if "country" in df.columns and not df["country"].isna().all()
            else None)
    df["is_foreign"] = ((df["country"] != home).astype(int) if home is not None
                        else 0)
    df["is_online"] = (df["channel"] == "online").astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)

    # How unusual is this amount *for its own category*? A $200 coffee is
    # bizarre; a $200 flight is cheap. Comparing against a single global average
    # would flag every plane ticket and miss every strange coffee.
    # Guard the divisor. If every transaction in a category costs the same --
    # three identical EGP 320 pharmacy charges, say -- the spread is zero and
    # this divides by it. The numerator is not exactly zero either, because
    # floating point leaves a speck behind, so the result is infinity rather
    # than the harmless NaN you might expect, and sklearn refuses to fit.
    # A category with no spread has no unusual amounts in it, so the honest
    # answer for every row there is zero.
    grp = df.groupby("category")["log_amount"]
    spread = grp.transform("std")
    df["amount_z_in_category"] = (
        (df["log_amount"] - grp.transform("mean")) / spread.where(spread > 0)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Velocity: how long since the previous transaction. Card testing fires
    # several charges within minutes, which almost never happens organically.
    gap = df["timestamp"].diff().dt.total_seconds().div(60)
    df["log_minutes_since_prev"] = np.log1p(gap.fillna(gap.max()).clip(lower=0))

    # Burst: how many transactions in the preceding hour.
    counter = pd.Series(1.0, index=pd.DatetimeIndex(df["timestamp"]))
    df["txns_last_hour"] = counter.rolling("1h").sum().to_numpy()

    if not has_time:
        # Neutralise the three clock-derived signals. Constant columns carry no
        # information, so the Isolation Forest simply ignores them instead of
        # splitting on an artefact of the export format.
        df["is_night"] = 0
        df["log_minutes_since_prev"] = 0.0
        df["txns_last_hour"] = 1.0

    # Size against this person's own baseline, not some global average.
    df["amount_vs_personal_median"] = df["amount"] / df["amount"].median()

    return df


def zscore_detector(df: pd.DataFrame, threshold: float = Z_THRESHOLD) -> np.ndarray:
    """Flag amounts more than `threshold` SDs above normal for their category."""
    return (df["amount_z_in_category"] > threshold).astype(int).to_numpy()


def build_isolation_forest() -> Pipeline:
    """
    Isolation Forest, in one paragraph.

    It builds random trees that split the data on random features at random
    cut points, then measures how many splits it takes to isolate each row.
    Ordinary rows sit in dense crowds and need many splits to separate. Odd rows
    sit alone and fall out after just a few. That average "isolation depth" is
    the anomaly score -- no labels required, and no assumption that the data is
    bell-shaped, which is where the Z-score approach breaks down.

    StandardScaler comes first because the features live on wildly different
    scales (hour 0-23, amount ratio 0-40). Without it the large-numbered
    features would dominate every split.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=300,
            contamination=CONTAMINATION,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def explain_anomaly(row: pd.Series) -> list[str]:
    """
    Turn a flagged row into reasons a human can argue with.

    A bare anomaly score of -0.63 is useless to a user and useless in a demo.
    The agent in Sprint 5 will read these strings straight out.
    """
    timed = bool(row.get("has_time", 1))

    reasons = []
    if row["is_foreign"]:
        reasons.append(f"foreign transaction ({row['city']}, {row['country']})")
    if timed and row["is_night"]:
        reasons.append(f"unusual hour ({int(row['hour']):02d}:00)")
    if row["amount_z_in_category"] > 2.5:
        reasons.append(f"amount {row['amount_z_in_category']:.1f} SDs above "
                       f"normal for {row['category']}")
    if row["amount_z_in_category"] < -2.0:
        reasons.append("unusually small amount (typical of card testing)")
    if timed and row["txns_last_hour"] >= 4:
        reasons.append(f"{int(row['txns_last_hour'])} transactions within an hour")
    if timed and row["log_minutes_since_prev"] < np.log1p(6):
        gap = np.expm1(row["log_minutes_since_prev"])
        reasons.append(f"only {gap:.0f} min after the previous charge")
    if row["amount_vs_personal_median"] > 8:
        reasons.append(f"{row['amount_vs_personal_median']:.0f}x this account's "
                       f"typical transaction")
    return reasons or ["unusual combination of amount, timing and channel"]


def score_block(name: str, y_true, y_pred, scores=None) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    out = {"method": name, "flagged": int(y_pred.sum()),
           "precision": p, "recall": r, "f1": f1}
    if scores is not None:
        out["pr_auc"] = average_precision_score(y_true, scores)
    return out


def main() -> None:
    df = add_fraud_features(spending_only(load_transactions()))
    y = df["is_fraud"].to_numpy()

    print("=" * 74)
    print("THE PROBLEM")
    print("=" * 74)
    print(f"Transactions : {len(df):,}")
    print(f"Fraud        : {y.sum()} ({y.mean():.2%})")
    print(f"\nA model that predicts 'never fraud' for every single row scores "
          f"{1 - y.mean():.2%} accuracy")
    print("and catches nothing. This is why accuracy is not reported below.")
    print("\nWhat we report instead:")
    print("  precision -- of the rows we flagged, how many were really fraud")
    print("               (low precision = the user drowns in false alarms)")
    print("  recall    -- of all real fraud, how much we caught")
    print("               (low recall = fraud goes through)")
    print("  PR-AUC    -- quality of the ranking, across every threshold")

    results = []

    # --- Baseline: statistical outliers on amount alone ---
    z_pred = zscore_detector(df)
    results.append(score_block(f"Z-score (>{Z_THRESHOLD} SD)", y, z_pred,
                               df["amount_z_in_category"]))

    # --- Isolation Forest across all features ---
    pipe = build_isolation_forest()
    X = df[FRAUD_FEATURES]
    pipe.fit(X)                                   # note: no y. Unsupervised.
    iso_pred = (pipe.predict(X) == -1).astype(int)
    iso_scores = -pipe.decision_function(X)       # higher = more anomalous
    results.append(score_block("Isolation Forest", y, iso_pred, iso_scores))

    print("\n" + "=" * 74)
    print("DETECTOR COMPARISON")
    print("=" * 74)
    table = pd.DataFrame(results).set_index("method")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    # --- What each fraud shape looks like to the detector ---
    df["iso_flag"] = iso_pred
    df["iso_score"] = iso_scores
    print("\n" + "=" * 74)
    print("RECALL BY FRAUD SIZE")
    print("=" * 74)
    fraud = df[df["is_fraud"] == 1].copy()
    bucket = pd.cut(fraud["amount"], [0, 60, 1000, 8000, 1_000_000],
                    labels=["<EGP 60 (card testing)", "EGP 60-1,000",
                            "EGP 1,000-8,000", ">EGP 8,000"])
    caught = fraud.groupby(bucket, observed=False)["iso_flag"].agg(["sum", "size"])
    caught["recall"] = (caught["sum"] / caught["size"]).round(2)
    print(caught.to_string())

    # --- Top flagged transactions, with human-readable reasons ---
    print("\n" + "=" * 74)
    print("TOP 8 FLAGGED TRANSACTIONS")
    print("=" * 74)
    top = df.nlargest(8, "iso_score")
    for _, row in top.iterrows():
        verdict = "FRAUD" if row["is_fraud"] else "legit"
        print(f"\n[{verdict:5s}] {row['timestamp']:%Y-%m-%d %H:%M}  "
              f"EGP {row['amount']:>8,.2f}  {row['description'][:38]}")
        for reason in explain_anomaly(row):
            print(f"          - {reason}")

    joblib.dump({"pipeline": pipe, "features": FRAUD_FEATURES}, MODEL_PATH)
    print(f"\n\nSaved Isolation Forest -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
