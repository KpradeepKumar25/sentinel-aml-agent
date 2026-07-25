"""
anomaly_detection.py

Hybrid anomaly detection: a fast, explainable rule-based layer plus the
trained Isolation Forest model exported from Colab. This mirrors exactly
what was validated in sentinel_aml_eda_training.ipynb, including two fixes
found during evaluation:

  1. The errorBalanceDest rule is only meaningful for non-merchant
     recipients, since PaySim never tracks real balances for merchant
     accounts (nameDest starting with 'M') -- those are always 0/0.
  2. Fraud in PaySim only occurs in TRANSFER and CASH_OUT transactions, so
     rules are restricted to those types to avoid flooding legitimate
     PAYMENT transactions with false positives.
  3. The rule engine flags on fullBalanceDrainAnomaly (origin balance fully
     emptied by the transaction), not origZeroBalanceAnomaly -- the latter
     ("oldbalanceOrg==0 & amount>0") measured against isFraud at 0.002%
     precision (25 true positives out of 1.3M flags) and is kept only
     because it's one of the pretrained model's feature_cols.pkl inputs.
     fullBalanceDrainAnomaly checks the actual documented PaySim fraud
     signature and measured at 100% precision / 97.5% recall.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _is_relevant_type(row) -> bool:
    """Handles both raw 'type' string columns and one-hot encoded columns,
    since feature_engineering.py returns different shapes depending on scope."""
    if "type" in row:
        return row["type"] in ("TRANSFER", "CASH_OUT")
    return bool(row.get("type_TRANSFER", 0) == 1 or row.get("type_CASH_OUT", 0) == 1)


def rule_based_flag(row, pattern: str = None) -> list:
    """Returns a list of human-readable reasons this row was flagged by rules.
    Empty list = not flagged.

    When pattern="structuring", only the actual structuring condition
    (near-threshold amount + repeat sender velocity) is checked -- unrelated
    balance-inconsistency flags are intentionally excluded so a "structuring"
    query never labels a non-structuring signal as structuring.
    """
    reasons = []

    if not _is_relevant_type(row):
        return reasons  # fraud never occurs outside TRANSFER/CASH_OUT in PaySim

    if row.get("nearThreshold10k", 0) == 1 and row.get("senderVelocity", 0) >= 3:
        reasons.append(
            "Multiple transactions just under the $10,000 reporting threshold "
            "(possible structuring)"
        )

    if pattern == "structuring":
        return reasons

    if row.get("fullBalanceDrainAnomaly", 0) == 1:
        reasons.append(
            "Origin account balance fully drained by this transaction "
            "(amount matches prior balance, new balance is zero)"
        )

    return reasons


def apply_rules(df: pd.DataFrame, pattern: str = None) -> pd.DataFrame:
    data = df.copy()
    data["rule_flags"] = data.apply(rule_based_flag, axis=1, pattern=pattern)
    data["rule_flagged"] = data["rule_flags"].apply(lambda x: len(x) > 0)
    return data


def load_model_artifacts():
    """Loads the trained Isolation Forest, scaler, and expected feature
    column order exported from the Colab notebook."""
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return model, scaler, feature_cols


def apply_ml_detection(df: pd.DataFrame, model=None, scaler=None, feature_cols=None) -> pd.DataFrame:
    """Scores rows with the trained Isolation Forest. If a required feature
    column is missing (e.g. because scope='structuring' returned a reduced
    column set), it's filled with 0 so the model can still run."""
    if model is None or scaler is None or feature_cols is None:
        model, scaler, feature_cols = load_model_artifacts()

    data = df.copy()

    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0

    X = data[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)

    data["ml_flagged"] = (model.predict(X_scaled) == -1).astype(int)
    data["ml_anomaly_score"] = -model.score_samples(X_scaled)

    return data


def detect(df: pd.DataFrame, mode: str = "hybrid", pattern: str = None) -> pd.DataFrame:
    """
    Main entry point used by executor.py.
      mode="hybrid"    -> rules + ML, flags the union of both (both catch the same
                          signal here, so this mode isn't used by any query type in
                          practice -- kept for completeness/future use)
      mode="rules_only" -> skip the ML model entirely. Used by structuring (the only
                          validated rule signature) and broad_analysis (the
                          full-balance-drain rule alone hits 100% precision / 97.5%
                          recall against isFraud -- adding ML or errorBalanceDest
                          drops precision to ~1.7% for a marginal recall gain, so
                          they're excluded here)
      mode="ml_only"    -> flags rows the ML model catches that the rule engine
                          does NOT -- i.e. the model's incremental value over the
                          validated rule, for queries that explicitly ask what
                          rules might miss
    """
    data = apply_rules(df, pattern=pattern)

    if mode == "rules_only":
        data["ml_flagged"] = 0
        data["ml_anomaly_score"] = 0.0
        data["hybrid_flagged"] = data["rule_flagged"]
        return data

    data = apply_ml_detection(data)

    if mode == "ml_only":
        data["hybrid_flagged"] = ((data["ml_flagged"] == 1) & (~data["rule_flagged"])).astype(bool)
        return data

    data["hybrid_flagged"] = (data["rule_flagged"] | (data["ml_flagged"] == 1)).astype(bool)
    return data