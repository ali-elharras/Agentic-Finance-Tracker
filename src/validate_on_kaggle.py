"""
Validate the fraud-detection approach on real, labelled fraud data.

    python -m src.validate_on_kaggle

Why this file exists
--------------------
Everything else in this project runs on data we generated ourselves. That is a
fair criticism of any synthetic-data project: of course the model finds the
fraud, we put it there. This script answers that objection by running the same
unsupervised technique against 284,807 real card transactions from European
cardholders (ULB / Kaggle), where 492 are confirmed fraud.

The dataset cannot be used for the rest of the product. Its features are V1-V28,
the output of a PCA transform the bank applied before release -- there are no
merchants, no categories, no real timestamps. You cannot ask "where am I
overspending?" of a matrix of anonymised principal components. It is a fraud
benchmark and nothing else, which is precisely how it is used here.

What gets measured
------------------
* Isolation Forest, unsupervised -- the same algorithm as Sprint 3, no labels.
* Logistic Regression, supervised -- an upper bound showing what labels buy you.
* The accuracy trap, ROC-AUC vs PR-AUC, and precision@k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/creditcard.csv"
TRAIN_FRAC = 0.70
SEED = 42


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Of the k most suspicious transactions, what share were really fraud?

    This is the metric a fraud team actually lives with. Analysts can review a
    fixed number of alerts per shift -- say the top 100. Precision@100 tells you
    how much of that shift is wasted on false alarms, which no threshold-free
    score like ROC-AUC ever communicates.
    """
    top = np.argsort(scores)[::-1][:k]
    return float(y_true[top].mean())


def report(name: str, y_true, y_pred, scores) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {
        "method": name,
        "flagged": int(y_pred.sum()),
        "precision": p,
        "recall": r,
        "f1": f1,
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "prec@100": precision_at_k(y_true, scores, 100),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Split chronologically on Time, matching Sprint 2's reasoning: the test set
    # must be the future. Fraud patterns drift, so a random split would let the
    # model learn from attacks it is then scored on.
    df = df.sort_values("Time").reset_index(drop=True)
    cut = int(len(df) * TRAIN_FRAC)
    train, test = df.iloc[:cut], df.iloc[cut:]

    feature_cols = [c for c in df.columns if c != "Class"]
    scaler = StandardScaler().fit(train[feature_cols])
    X_train = scaler.transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    y_train = train["Class"].to_numpy()
    y_test = test["Class"].to_numpy()

    print("=" * 78)
    print("REAL FRAUD DATA -- ULB / Kaggle credit card dataset")
    print("=" * 78)
    print(f"Total transactions : {len(df):,}")
    print(f"Confirmed fraud    : {int(df.Class.sum())} ({df.Class.mean():.4%})")
    print(f"Train / test       : {len(train):,} / {len(test):,} "
          f"(chronological split)")
    print(f"Fraud in test      : {int(y_test.sum())}")

    print("\n" + "-" * 78)
    print("THE ACCURACY TRAP")
    print("-" * 78)
    naive = np.zeros_like(y_test)
    print(f"Predict 'never fraud' for every row -> accuracy "
          f"{accuracy_score(y_test, naive):.4%}")
    print("It catches zero fraud. Any report quoting accuracy on this dataset")
    print("is either misleading its reader or hiding a broken model.")

    results = []

    # ---------------------------------------------------------------
    # 1. Isolation Forest -- unsupervised, exactly as in Sprint 3
    # ---------------------------------------------------------------
    # contamination is our stated prior for how much of the data is anomalous.
    # Setting it near the true fraud rate is generous; in production you would
    # be guessing. The PR-AUC column is threshold-free and does not depend on it.
    iso = IsolationForest(
        n_estimators=200, contamination=0.002,
        random_state=SEED, n_jobs=-1,
    ).fit(X_train)                       # note: y_train is NOT passed

    iso_scores = -iso.decision_function(X_test)
    iso_pred = (iso.predict(X_test) == -1).astype(int)
    results.append(report("Isolation Forest (unsupervised)",
                          y_test, iso_pred, iso_scores))

    # ---------------------------------------------------------------
    # 2. Isolation Forest trained on confirmed-legitimate rows only
    # ---------------------------------------------------------------
    # A middle path between unsupervised and supervised, called novelty
    # detection. We never show the model a single fraud example -- we simply
    # refuse to let fraud pollute its idea of "normal". Any bank has years of
    # confirmed-good transactions, so this is realistic, and it fixes a real
    # weakness: fitting on contaminated data teaches the forest that a bit of
    # fraud is part of the normal crowd.
    iso_clean = IsolationForest(
        n_estimators=200, contamination=0.002,
        random_state=SEED, n_jobs=-1,
    ).fit(X_train[y_train == 0])

    iso_clean_scores = -iso_clean.decision_function(X_test)
    iso_clean_pred = (iso_clean.predict(X_test) == -1).astype(int)
    results.append(report("Isolation Forest (legit-only fit)",
                          y_test, iso_clean_pred, iso_clean_scores))

    # ---------------------------------------------------------------
    # 3. Logistic Regression -- supervised upper bound
    # ---------------------------------------------------------------
    # class_weight="balanced" makes each of the ~350 training frauds count as
    # much as ~570 legitimate rows. Without it the model minimises total error
    # by predicting "legit" every time and learns nothing at all.
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED,
    ).fit(X_train, y_train)

    lr_scores = lr.predict_proba(X_test)[:, 1]
    lr_pred = lr.predict(X_test)
    results.append(report("Logistic Regression (supervised)",
                          y_test, lr_pred, lr_scores))

    print("\n" + "=" * 78)
    print("RESULTS ON REAL DATA")
    print("=" * 78)
    table = pd.DataFrame(results).set_index("method")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n" + "-" * 78)
    print("ROC-AUC vs PR-AUC")
    print("-" * 78)
    print("ROC-AUC looks excellent for both models. Ignore it here.")
    print("With 99.83% of rows negative, a model can rank almost everything")
    print("correctly and still produce alerts that are mostly false alarms.")
    print("PR-AUC and precision@100 describe what the analyst actually sees.")

    print("\n" + "-" * 78)
    print("HOW MANY ALERTS WOULD A HUMAN REVIEW?")
    print("-" * 78)
    print(f"{'':>9}{'unsupervised':>14}{'legit-only':>14}{'supervised':>14}")
    for k in (50, 100, 500):
        print(f"top {k:>4}:"
              f"{precision_at_k(y_test, iso_scores, k):>13.1%}"
              f"{precision_at_k(y_test, iso_clean_scores, k):>14.1%}"
              f"{precision_at_k(y_test, lr_scores, k):>14.1%}")


if __name__ == "__main__":
    main()
