"""
add_synthetic_structuring_demo.py

Adds a SMALL, CLEARLY-DISCLOSED block of synthetic transactions to the demo
sample dataset, simulating a repeat sender performing classic structuring
(many transactions just under the $10,000 reporting threshold).

WHY THIS EXISTS (put this in your README too):
PaySim's real customer IDs are essentially single-use (max ~3 transactions
per sender across the entire 6.3M-row dataset), so the "same sender, many
transactions" pattern that defines structuring/smurfing cannot occur
naturally in this data. Rather than fabricating or editing real transaction
records, this script APPENDS a clearly-labeled synthetic block so the
structuring and aggregation detection pathways can be demonstrated honestly.

This is disclosed, rule-compliant use of synthetic data per the hackathon's
own rules: "Teams may use synthetic data ... schema, field definitions,
assumptions, and data generation logic must be relevant to the use case and
clearly documented in the README.md file."

The synthetic sender ID is intentionally recognizable: C999000001
"""

import pandas as pd
import numpy as np

SAMPLE_PATH = "data/raw/paysim_sample.csv"
SYNTHETIC_SENDER_ID = "C999000001"
N_TRANSACTIONS = 12  # >10, to trigger both the structuring rule and the
                       # "10+ transactions under $10,000" aggregation query

np.random.seed(7)


def build_synthetic_block(df: pd.DataFrame) -> pd.DataFrame:
    max_step = df["step"].max()
    min_step = df["step"].min()

    # Spread transactions across the last ~20 days so they fall inside a
    # "last 30 days" filter with some margin -- clamped so it never goes
    # below the dataset's actual minimum step
    window_start = max(min_step, max_step - 20 * 24)
    if window_start >= max_step:
        window_start = max_step - N_TRANSACTIONS  # tiny fallback for very small datasets

    available_range = max_step - window_start
    if available_range < N_TRANSACTIONS:
        # not enough distinct integer steps in range -- allow repeats
        steps = sorted(np.random.choice(range(int(window_start), int(max_step) + 1), N_TRANSACTIONS, replace=True))
    else:
        steps = sorted(np.random.choice(range(int(window_start), int(max_step) + 1), N_TRANSACTIONS, replace=False))

    amounts = np.random.uniform(9000, 9950, N_TRANSACTIONS).round(2)

    # Give the sender a starting balance large enough to make these
    # transactions plausible (not a full-balance-drain, since structuring
    # is a DIFFERENT pattern from the drain-signature fraud rule -- keeping
    # them distinct demonstrates the two rules catch different things)
    starting_balance = 150_000.0
    balance = starting_balance

    rows = []
    for i in range(N_TRANSACTIONS):
        old_bal = balance
        new_bal = round(old_bal - amounts[i], 2)
        balance = new_bal

        receiver_id = f"C888{str(i).zfill(6)}"  # distinct receivers, same sender

        rows.append({
            "step": steps[i],
            "type": "TRANSFER",
            "amount": amounts[i],
            "nameOrig": SYNTHETIC_SENDER_ID,
            "oldbalanceOrg": old_bal,
            "newbalanceOrig": new_bal,
            "nameDest": receiver_id,
            "oldbalanceDest": 0.0,
            "newbalanceDest": amounts[i],
            "isFraud": 0,          # intentionally NOT labeled as fraud --
                                    # structuring often evades naive fraud
                                    # labels, which is part of the point
            "isFlaggedFraud": 0,
        })

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(SAMPLE_PATH)
    print("Existing sample:", df.shape)

    if (df["nameOrig"] == SYNTHETIC_SENDER_ID).any():
        print(f"Synthetic sender {SYNTHETIC_SENDER_ID} already present -- skipping, nothing to add.")
        return

    synthetic_block = build_synthetic_block(df)
    print("\nSynthetic block preview:")
    print(synthetic_block[["step", "amount", "nameOrig", "nameDest"]])

    combined = pd.concat([df, synthetic_block], ignore_index=True)
    combined.to_csv(SAMPLE_PATH, index=False)

    print(f"\nSaved. New sample shape: {combined.shape}")
    print(f"Synthetic rows added: {len(synthetic_block)}")
    print(f"\nDemo queries this enables:")
    print(f'  "Find structuring patterns in the last 30 days"')
    print(f'  "Which customers made 10+ transactions under $10,000?"')
    print(f'  "Is customer ID {SYNTHETIC_SENDER_ID} suspicious?"')


if __name__ == "__main__":
    main()
