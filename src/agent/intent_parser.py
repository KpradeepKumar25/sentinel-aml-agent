"""
intent_parser.py

Parses a natural-language user query into a structured intent object.
This is Phase 2, Step 1: keyword/rule-based parsing (fast, explainable, no API needed).

Upgrade path (optional, only if time allows): swap the keyword matching in
`parse_query()` for an LLM call that returns the same QueryIntent structure -
the rest of the agent (planner, executor) doesn't need to change at all.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryIntent:
    """Structured representation of what the user is asking for."""
    raw_query: str
    query_type: str                      # "broad_analysis" | "structuring" | "aggregation" | "single_entity" | "ml_anomaly"
    target_pattern: Optional[str] = None # e.g. "structuring", "smurfing", "layering"
    customer_id: Optional[str] = None
    date_range_days: Optional[int] = None
    transaction_type: Optional[str] = None   # e.g. "TRANSFER", "CASH_OUT"
    amount_threshold: Optional[float] = None
    min_transaction_count: Optional[int] = None
    needs_full_eda: bool = False
    needs_ml_detection: bool = True
    tags: list = field(default_factory=list)  # human-readable trail of what was detected, for the "why" output


# --- Keyword banks used for classification ---
STRUCTURING_KEYWORDS = ["structuring", "smurf", "smurfing", "split", "layering"]
SINGLE_ENTITY_PATTERNS = [
    r"customer\s*(?:id)?\s*[:#]?\s*([A-Za-z]?\d+)",
    r"account\s*(?:id)?\s*[:#]?\s*([A-Za-z]?\d+)",
    r"\bid\s*([A-Za-z]?\d+)\b",
]
AGGREGATION_KEYWORDS = ["how many", "which customers", "count", "average", "total", "made \\d+"]
BROAD_KEYWORDS = ["analyse", "analyze", "scan", "review the dataset", "overall", "full report", "everything"]
ML_ANOMALY_KEYWORDS = [
    "rules might miss", "rule might miss", "rules would miss", "rules can't catch",
    "rules can not catch", "without a known rule", "no known rule", "ml only",
    "machine learning", "isolation forest", "anomalous transactions",
]

TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]

# Broad, deliberately generous AML/finance vocabulary check used only to catch
# queries with NO domain signal at all (e.g. "hi", "what's the weather") before
# they'd otherwise silently fall through to the broad-analysis default. Doesn't
# need to be precise -- a query only needs to match ANY one of these to be
# treated as in-scope; being over-inclusive here is the safe failure mode.
DOMAIN_KEYWORDS = [
    "transaction", "customer", "account", "suspicious", "fraud", "launder",
    "money", "structuring", "smurf", "layering", "anomaly", "anomalous",
    "pattern", "flag", "risk", "transfer", "payment", "analyse", "analyze",
    "dataset", "data", "detect", "activity", "sar", "report", "threshold",
    "amount", "bank", "financial", "finance", "wire", "deposit", "withdraw",
    "cash", "balance", "sender", "receiver", "compliance", "aml",
]


def _extract_date_range(query: str) -> Optional[int]:
    """Looks for phrases like 'last 30 days', 'past 7 days', 'last week/month'."""
    match = re.search(r"last\s+(\d+)\s+day", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if re.search(r"last\s+week", query, re.IGNORECASE):
        return 7
    if re.search(r"last\s+month", query, re.IGNORECASE):
        return 30
    return None


def _extract_customer_id(query: str) -> Optional[str]:
    for pattern in SINGLE_ENTITY_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_amount_threshold(query: str) -> Optional[float]:
    """Looks for phrases like 'under $10,000' or 'over 5000'. Prioritizes
    dollar-prefixed numbers so a stray '10+' (meaning "10 or more transactions")
    doesn't get mistaken for a dollar amount."""
    # Prefer an explicit $-prefixed number first (most reliable signal)
    dollar_match = re.search(r"\$\s*([\d,]+)(?:\.\d+)?", query)
    if dollar_match:
        cleaned = dollar_match.group(1).replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    # Fall back to a number near threshold-ish words, but require 3+ digits
    # so we don't pick up small counts like "10+ transactions"
    if any(w in query.lower() for w in ["under", "below", "over", "above"]):
        match = re.search(r"\b(\d{3,}(?:,\d{3})*)\b", query)
        if match:
            cleaned = match.group(1).replace(",", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
    return None


def _extract_transaction_count(query: str) -> Optional[int]:
    """Looks for phrases like '10+ transactions' or 'made 10 transactions'."""
    match = re.search(r"(\d+)\s*\+?\s*transactions?", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_transaction_type(query: str) -> Optional[str]:
    for t in TRANSACTION_TYPES:
        if t.lower() in query.lower():
            return t
    return None


def parse_query(query: str) -> QueryIntent:
    """
    Main entry point. Takes a raw natural-language query and returns a
    structured QueryIntent that the planner will use to decide which
    tools to invoke.
    """
    q_lower = query.lower()
    tags = []

    customer_id = _extract_customer_id(query)
    date_range_days = _extract_date_range(query)
    transaction_type = _extract_transaction_type(query)
    amount_threshold = _extract_amount_threshold(query)
    min_transaction_count = _extract_transaction_count(query)

    # --- Decide the primary query type (priority order matters) ---

    # 1. Single-entity lookup takes priority over everything else
    if customer_id:
        tags.append(f"Detected single-entity query for customer/account ID {customer_id}")
        return QueryIntent(
            raw_query=query,
            query_type="single_entity",
            customer_id=customer_id,
            date_range_days=date_range_days,
            transaction_type=transaction_type,
            needs_full_eda=False,
            needs_ml_detection=False,   # per spec: explain existing flags or compute on-demand only
            tags=tags,
        )

    # 2. Structuring / pattern-specific query
    if any(kw in q_lower for kw in STRUCTURING_KEYWORDS):
        tags.append("Detected structuring/smurfing pattern keyword")
        if date_range_days:
            tags.append(f"Applying date filter: last {date_range_days} days")
        return QueryIntent(
            raw_query=query,
            query_type="structuring",
            target_pattern="structuring",
            date_range_days=date_range_days,
            transaction_type=transaction_type,
            needs_full_eda=False,       # skip full EDA per spec example
            needs_ml_detection=True,
            tags=tags,
        )

    # 3. Direct aggregation / threshold query (no ML needed)
    if any(re.search(kw, q_lower) for kw in AGGREGATION_KEYWORDS) or amount_threshold:
        tags.append("Detected aggregation/threshold-style query")
        if amount_threshold:
            tags.append(f"Extracted amount threshold: {amount_threshold}")
        return QueryIntent(
            raw_query=query,
            query_type="aggregation",
            amount_threshold=amount_threshold,
            min_transaction_count=min_transaction_count,
            transaction_type=transaction_type,
            date_range_days=date_range_days,
            needs_full_eda=False,
            needs_ml_detection=False,   # per spec: run aggregation/rule directly, ML not required
            tags=tags,
        )

    # 4. ML-anomaly query -- explicitly asks for patterns without a known rule
    #    signature, i.e. what the Isolation Forest can catch beyond the
    #    validated rule set.
    if any(kw in q_lower for kw in ML_ANOMALY_KEYWORDS):
        tags.append("Detected ML-anomaly query (patterns without a known rule signature)")
        if date_range_days:
            tags.append(f"Applying date filter: last {date_range_days} days")
        return QueryIntent(
            raw_query=query,
            query_type="ml_anomaly",
            date_range_days=date_range_days,
            transaction_type=transaction_type,
            needs_full_eda=False,
            needs_ml_detection=True,
            tags=tags,
        )

    # 5. No specific pattern matched AND no recognizable AML/finance vocabulary
    #    at all -- rather than silently defaulting to a full broad-analysis run
    #    (which would misleadingly imply the query was understood), say plainly
    #    that it's out of scope instead of running the pipeline on it.
    if not any(kw in q_lower for kw in DOMAIN_KEYWORDS):
        tags.append("Query does not appear related to suspicious activity detection")
        return QueryIntent(
            raw_query=query,
            query_type="unrelated",
            needs_full_eda=False,
            needs_ml_detection=False,
            tags=tags,
        )

    # 6. Broad / full-dataset analysis (default / fallback)
    tags.append("No specific entity or pattern detected - treating as broad analysis")
    return QueryIntent(
        raw_query=query,
        query_type="broad_analysis",
        date_range_days=date_range_days,
        transaction_type=transaction_type,
        needs_full_eda=True,
        needs_ml_detection=True,
        tags=tags,
    )


if __name__ == "__main__":
    # Quick manual test against the exact example queries from the problem statement
    test_queries = [
        "Find structuring patterns in the last 30 days",
        "Which customers made 10+ transactions under $10,000?",
        "Is customer ID C67886069 suspicious?",
        "Analyse this dataset for suspicious activity",
    ]
    for q in test_queries:
        intent = parse_query(q)
        print(f"\nQuery: {q}")
        print(intent)