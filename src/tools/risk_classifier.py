"""
risk_classifier.py

Converts anomaly scores + rule signals into a business-friendly risk tier
(LOW / MEDIUM / HIGH / NONE). Vectorized with np.select for speed on large
datasets -- the row-by-row .apply() version was ~100x slower during
Colab testing on 200k+ rows.
"""

import numpy as np
import pandas as pd


def classify_risk(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "ml_anomaly_score" not in data.columns:
        data["ml_anomaly_score"] = 0.0
    if "rule_flagged" not in data.columns:
        data["rule_flagged"] = False

    # rules_only mode (no ML score computed) leaves ml_anomaly_score at a
    # constant 0.0 for every row. The quantile logic below can't
    # differentiate against a constant column -- q90 == q75 == 0, so the
    # HIGH tier becomes mathematically unreachable and every rule hit gets
    # silently downgraded to MEDIUM. Since the rule flagged here (the
    # full-balance-drain signature) is validated at 100% precision, a hit
    # on it alone is high-confidence evidence, not a "maybe" -- classify it
    # as HIGH directly instead of routing it through ML-score quantiles
    # that don't exist for this mode.
    if not (data["ml_anomaly_score"] != 0).any():
        data["risk_level"] = np.where(data["rule_flagged"], "HIGH", "NONE")
        return data

    # Guard against tiny slices (e.g. single_entity queries) where quantiles
    # aren't meaningful -- fall back to fixed thresholds in that case.
    if len(data) >= 20:
        q90 = data["ml_anomaly_score"].quantile(0.90)
        q75 = data["ml_anomaly_score"].quantile(0.75)
    else:
        q90 = data["ml_anomaly_score"].max() if len(data) else 0
        q75 = data["ml_anomaly_score"].median() if len(data) else 0

    conditions = [
        (data["rule_flagged"]) & (data["ml_anomaly_score"] > q90),
        (data["rule_flagged"]) | (data["ml_anomaly_score"] > q90),
        (data["ml_anomaly_score"] > q75),
    ]
    choices = ["HIGH", "MEDIUM", "LOW"]

    data["risk_level"] = np.select(conditions, choices, default="NONE")
    return data


def suggest_action(risk_level: str) -> str:
    """Maps a risk tier to the recommended escalation action."""
    return {
        "HIGH": "report",
        "MEDIUM": "flag for review",
        "LOW": "monitor",
        "NONE": "no action",
    }.get(risk_level, "monitor")