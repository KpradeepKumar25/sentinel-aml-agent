"""
data_loader.py

Loads the PaySim dataset and applies whatever filters the planner decided
are relevant for this query (customer ID, date range, transaction type).
Loading only what's needed (instead of always loading everything) is part
of what makes the agent "query-aware" rather than a fixed pipeline.
"""

import os
import pandas as pd

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def find_csv(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Finds the PaySim CSV inside the raw data folder."""
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            return os.path.join(data_dir, f)
    raise FileNotFoundError(
        f"No CSV found in {data_dir}. Download PaySim and place it there, "
        f"or pass an explicit path to load_data()."
    )


def load_data(
    path: str = None,
    filter_customer_id: str = None,
    date_range_days: int = None,
    transaction_type: str = None,
) -> pd.DataFrame:
    """
    Loads the dataset and applies filters selectively.

    PaySim's 'step' column represents simulated hours (1 step = 1 hour),
    so a "last N days" filter is translated into the last N*24 steps
    relative to the max step present in the data.
    """
    if path is None:
        path = find_csv()

    df = pd.read_csv(path)

    if filter_customer_id:
        target_id = str(filter_customer_id)
        df = df[
            (df["nameOrig"].astype(str) == target_id)
            | (df["nameDest"].astype(str) == target_id)
        ]

    if date_range_days:
        max_step = df["step"].max()
        cutoff_step = max_step - (date_range_days * 24)
        df = df[df["step"] >= cutoff_step]

    if transaction_type:
        df = df[df["type"] == transaction_type]

    return df.reset_index(drop=True)