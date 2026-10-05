"""
Produce a realistic bank export for the upload demo.

    python -m src.export_sample

Writes `data/processed/sample_bank_export.csv` -- the generated transactions
stripped back to what a bank actually gives you:

* capitalised headers ("Date", "Description") like a real download
* signed amounts, negative for money out
* NO `category` column
* NO `is_fraud` column
* NO clean `merchant` name, only the raw descriptor

Those omissions are the whole point. Uploading this file forces the categoriser
to predict all eight categories from raw text, and forces the anomaly detector
to work with no answer key. A demo that uploads data already containing the
answers proves nothing.
"""

from __future__ import annotations

import pandas as pd

from src.features import load_transactions

OUT_PATH = "data/processed/sample_bank_export.csv"


def main() -> None:
    df = load_transactions()

    export = pd.DataFrame({
        "Date": df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Description": df["description"],
        # Signed, the way every bank export does it: negative is money leaving.
        "Amount": [
            -abs(a) if d == "debit" else abs(a)
            for a, d in zip(df["amount"], df["direction"])
        ],
        "Channel": df["channel"],
        "City": df["city"],
        "Country": df["country"],
    })

    export.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(export):,} rows to {OUT_PATH}")
    print(f"Columns  : {', '.join(export.columns)}")
    print(f"Withheld : category, is_fraud, merchant")
    print("\nFirst 3 rows:")
    print(export.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
