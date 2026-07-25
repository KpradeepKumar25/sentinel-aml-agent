"""
build_sample.py

One-time utility: builds a smaller PaySim sample for fast local iteration.
Preserves every fraud row (so precision/recall checks against isFraud stay
meaningful) and guarantees the demo customer ID used throughout the README
and test queries is present in the sample.

Run from the project root:
    python build_sample.py

Writes ./paysim_sample.csv -- copy it into data/raw/ afterward (see README).
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paysim_sample.csv")
DEMO_CUSTOMER_ID = "C67886069"
TARGET_ROWS = 33000
RANDOM_STATE = 42


def find_full_csv() -> str:
    for f in os.listdir(RAW_DIR):
        if f.endswith(".csv"):
            return os.path.join(RAW_DIR, f)
    raise FileNotFoundError(f"No CSV found in {RAW_DIR}")


def main():
    src = find_full_csv()
    print(f"Loading full dataset from {src} ...")
    df = pd.read_csv(src)
    print(f"Full dataset: {len(df):,} rows, {df['isFraud'].sum():,} fraud rows")

    fraud = df[df["isFraud"] == 1]
    demo_rows = df[(df["nameOrig"] == DEMO_CUSTOMER_ID) | (df["nameDest"] == DEMO_CUSTOMER_ID)]
    non_fraud = df[df["isFraud"] == 0]
    legit_pool = non_fraud[~non_fraud.index.isin(demo_rows.index)]

    n_legit_needed = max(TARGET_ROWS - len(fraud) - len(demo_rows), 0)
    legit_sample = legit_pool.sample(n=min(n_legit_needed, len(legit_pool)), random_state=RANDOM_STATE)

    sample = pd.concat([fraud, demo_rows, legit_sample]).drop_duplicates()
    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle row order

    sample.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Sample written to {OUTPUT_PATH}")
    print(f"Sample rows: {len(sample):,}")
    print(f"Fraud rows kept: {sample['isFraud'].sum():,} (expected {fraud.shape[0]:,})")
    print(f"Demo customer '{DEMO_CUSTOMER_ID}' rows present: {len(demo_rows)}")


if __name__ == "__main__":
    main()
