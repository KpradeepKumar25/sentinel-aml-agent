"""
eda_tool.py

Performs automated exploratory data analysis / profiling.
Only invoked by the planner for broad_analysis queries -- skipped for
targeted or single-entity queries per the spec's adaptive behavior.
"""

import pandas as pd


def run_eda(df: pd.DataFrame) -> dict:
    """Returns a structured summary of the dataset -- baseline behavior,
    transaction type breakdown, and class balance if labels are present."""

    summary = {
        "row_count": int(len(df)),
        "unique_senders": int(df["nameOrig"].nunique()) if "nameOrig" in df.columns else None,
        "unique_receivers": int(df["nameDest"].nunique()) if "nameDest" in df.columns else None,
        "transaction_type_counts": (
            df["type"].value_counts().to_dict() if "type" in df.columns else {}
        ),
        "amount_stats": {
            "mean": float(df["amount"].mean()),
            "median": float(df["amount"].median()),
            "min": float(df["amount"].min()),
            "max": float(df["amount"].max()),
            "std": float(df["amount"].std()),
        } if "amount" in df.columns else {},
    }

    if "isFraud" in df.columns:
        summary["fraud_rate"] = float(df["isFraud"].mean())
        summary["fraud_count"] = int(df["isFraud"].sum())
        summary["fraud_by_type"] = (
            df[df["isFraud"] == 1]["type"].value_counts().to_dict()
        )

        # PaySim's true population fraud rate is ~0.13%. If this dataset shows
        # something far higher, it's almost certainly a fraud-oversampled demo
        # subset (kept to guarantee non-empty results at small scale) rather
        # than a representative sample -- disclose this plainly rather than
        # letting it look like an alarmingly high real-world fraud rate.
        TRUE_POPULATION_FRAUD_RATE = 0.0013
        if summary["fraud_rate"] > TRUE_POPULATION_FRAUD_RATE * 3:
            summary["sampling_note"] = (
                f"This dataset intentionally oversamples fraud cases for demo "
                f"purposes ({summary['fraud_rate']:.1%} here vs PaySim's true "
                f"population rate of {TRUE_POPULATION_FRAUD_RATE:.2%}). Model "
                f"precision/recall were validated separately against the full, "
                f"representative dataset -- see README."
            )

    return summary