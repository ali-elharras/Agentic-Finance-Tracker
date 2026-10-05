"""
Train the spending-category classifier.

    python -m src.train_categorizer

Compares Logistic Regression against a Random Forest on identical features,
reports honest metrics, and saves the better model to models/categorizer.pkl.

What the model is given
-----------------------
* the raw bank descriptor, as text        e.g. "POS DEBIT WALMART SUPERCE #6283"
* the amount, raw and log-scaled
* hour, weekday, day-of-month, weekend flag
* the channel (card_present / online / transfer)

What it is NOT given: the clean `merchant` column. See features.py for why.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (CATEGORICAL_COLS, NUMERIC_COLS, TEXT_COL,
                          label_ceiling, load_transactions, spending_only,
                          time_split)

MODEL_PATH = "models/categorizer.pkl"
SEED = 42


def build_preprocessor() -> ColumnTransformer:
    """
    Turn a row of mixed data into a row of numbers.

    Two text vectorisers run over the same descriptor, because they catch
    different things:

    * word n-grams learn whole tokens -- "STARBUCKS", "POS DEBIT", "AUTOPAY".
    * character n-grams learn fragments -- "STARB", "MZN M". These survive the
      truncation banks apply, so "BLUE BOTTLE COFFE" and "BLUE BOTTLE COFFEE"
      still look alike. Word n-grams treat them as unrelated.

    TF-IDF weighting means a token appearing in nearly every descriptor (like
    "DEBIT") counts for little, while a rare, distinctive one counts for a lot.
    """
    return ColumnTransformer([
        ("word", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_features=400,
            # Letters only: the digits in "#6283" are random reference numbers,
            # pure noise that would waste model capacity.
            token_pattern=r"[A-Za-z]{2,}",
        ), TEXT_COL),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=900,
        ), TEXT_COL),
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def build_models() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=3000, C=2.0, random_state=SEED)),
        ]),
        "RandomForest": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                random_state=SEED)),
        ]),
    }


def importance_by_block(pipe: Pipeline) -> pd.Series:
    """
    Group feature importances by where the feature came from.

    Individual TF-IDF importances are meaningless on their own -- there are
    1,300 of them. Rolled up per block they answer a question worth asking:
    is the model reading the descriptor, or leaning on amount and time?
    """
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        weights = clf.feature_importances_          # tree models
    else:
        # Linear models: average the absolute coefficient across all 8 classes.
        # Absolute, because a strongly negative coefficient ("this is NOT
        # Groceries") is just as informative as a positive one.
        weights = np.abs(clf.coef_).mean(axis=0)

    blocks = pd.Series(weights, index=[n.split("__")[0] for n in names])
    grouped = blocks.groupby(level=0).sum()
    return (grouped / grouped.sum() * 100).round(1).sort_values(ascending=False)


def top_terms(pipe: Pipeline, n: int = 6) -> pd.DataFrame:
    """The descriptor fragments pushing hardest toward each category."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame()
    names = pipe.named_steps["prep"].get_feature_names_out()
    keep = [i for i, nm in enumerate(names) if nm.startswith(("word__", "char__"))]
    rows = {}
    for ci, cls in enumerate(clf.classes_):
        coefs = clf.coef_[ci][keep]
        best = np.argsort(coefs)[::-1][:n]
        rows[cls] = [names[keep[b]].split("__", 1)[1].strip() for b in best]
    return pd.DataFrame(rows).T


def main() -> None:
    df = spending_only(load_transactions())
    train, test = time_split(df, test_fraction=0.2)

    y_train, y_test = train["category"], test["category"]

    print("=" * 74)
    print("DATA")
    print("=" * 74)
    print(f"Training rows : {len(train):,}  "
          f"({train['timestamp'].min():%Y-%m-%d} to {train['timestamp'].max():%Y-%m-%d})")
    print(f"Test rows     : {len(test):,}  "
          f"({test['timestamp'].min():%Y-%m-%d} to {test['timestamp'].max():%Y-%m-%d})")

    # --- The two reference points that make accuracy interpretable ---
    dummy = DummyClassifier(strategy="most_frequent").fit(train, y_train)
    floor = accuracy_score(y_test, dummy.predict(test))
    ceiling = label_ceiling(df)

    print(f"\nFloor    (always guess the most common category) : {floor:6.1%}")
    print(f"Ceiling~ (ambiguity + 3% label noise, pessimistic): {ceiling:6.1%}")
    print("The ceiling is an estimate and assumes one-off merchants are")
    print("unguessable, so a few points above it is healthy. Near 100% would")
    print("mean leakage, since 3% of the labels are deliberately wrong.")

    # --- Train and compare ---
    results = {}
    for name, pipe in build_models().items():
        pipe.fit(train, y_train)
        pred = pipe.predict(test)
        results[name] = {
            "pipe": pipe,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "pred": pred,
        }

    print("\n" + "=" * 74)
    print("MODEL COMPARISON")
    print("=" * 74)
    print(f"{'model':22s} {'accuracy':>10s} {'macro F1':>10s}")
    for name, r in results.items():
        print(f"{name:22s} {r['accuracy']:>9.1%} {r['macro_f1']:>10.3f}")

    # Macro F1 decides, not accuracy. Accuracy is dominated by the big
    # categories; macro F1 averages the score of every category equally, so a
    # model that ignores Travel entirely cannot hide behind Groceries.
    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best = results[best_name]
    print(f"\nWinner (by macro F1): {best_name}")

    print("\n" + "=" * 74)
    print(f"PER-CATEGORY BREAKDOWN -- {best_name}")
    print("=" * 74)
    print(classification_report(y_test, best["pred"], zero_division=0))

    print("=" * 74)
    print("CONFUSION MATRIX (rows = truth, columns = prediction)")
    print("=" * 74)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, best["pred"], labels=labels)
    short = [c[:9] for c in labels]
    print(pd.DataFrame(cm, index=short, columns=short).to_string())

    print("\n" + "=" * 74)
    print("WHERE THE MODEL LOOKS (% of total weight)")
    print("=" * 74)
    print(importance_by_block(best["pipe"]).to_string())

    terms = top_terms(best["pipe"])
    if not terms.empty:
        print("\n" + "=" * 74)
        print("STRONGEST DESCRIPTOR CLUES PER CATEGORY")
        print("=" * 74)
        print(terms.to_string(header=False))

    joblib.dump(best["pipe"], MODEL_PATH)
    print(f"\nSaved {best_name} -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
