"""
explanation_engine.py

Generates plain-English explanations for flagged transactions/customers,
tied back to the original query intent -- plus a lightweight "Suggested SAR
Report" draft, reframing the same explanation into a compliance-report style
summary (the differentiator feature from the README).
"""

import pandas as pd
from risk_classifier import suggest_action


def explain_row(row) -> str:
    """Builds a human-readable explanation from whatever signals fired."""
    parts = []

    reasons = row.get("rule_flags", [])
    if reasons:
        parts.append("; ".join(reasons))

    if row.get("ml_flagged", 0) == 1:
        score = row.get("ml_anomaly_score", 0)
        parts.append(f"Flagged by anomaly detection model (score: {score:.3f})")

    if not parts:
        return "No significant anomaly signals detected."

    return " | ".join(parts)


def draft_sar(row) -> str:
    """Reframes the explanation into a short SAR-style summary."""
    sender = row.get("nameOrig", "Unknown sender")
    receiver = row.get("nameDest", "Unknown receiver")
    amount = row.get("amount", 0)
    reasons = row.get("rule_flags", [])
    risk = row.get("risk_level", "UNKNOWN")

    reason_text = "; ".join(reasons) if reasons else "elevated anomaly score from model-based detection"

    return (
        f"SUSPICIOUS ACTIVITY SUMMARY -- Risk: {risk}\n"
        f"Party: {sender} -> {receiver} | Amount: ${amount:,.2f}\n"
        f"Basis for flag: {reason_text}.\n"
        f"Recommended action: {suggest_action(risk)}."
    )


def explain(df: pd.DataFrame, mode: str = "broad", customer_id: str = None) -> list:
    """
    Main entry point. Returns a list of dicts, one per flagged row, with
    explanation + SAR draft + recommended action -- the structured output
    format the hackathon spec asks judges to be able to inspect.
    """
    flagged_col = "hybrid_flagged" if "hybrid_flagged" in df.columns else "rule_flagged"
    flagged = df[df[flagged_col] == True].copy() if flagged_col in df.columns else df.copy()

    if "risk_level" not in flagged.columns:
        flagged["risk_level"] = "UNKNOWN"

    results = []
    for _, row in flagged.iterrows():
        results.append({
            "sender": row.get("nameOrig"),
            "receiver": row.get("nameDest"),
            "amount": float(row.get("amount", 0)),
            "risk_level": row.get("risk_level"),
            "explanation": explain_row(row),
            "suggested_sar": draft_sar(row),
            "recommended_action": suggest_action(row.get("risk_level", "LOW")),
        })

    return results