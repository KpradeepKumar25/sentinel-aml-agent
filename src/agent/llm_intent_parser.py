"""
llm_intent_parser.py

OPTIONAL enhancement layer. Uses a free-tier LLM (Groq, OpenAI-compatible API)
to parse the user's query into the same QueryIntent structure produced by
intent_parser.py's regex/keyword parser.

This is NOT a replacement for intent_parser.py -- it's an alternate path.
If the API key isn't set, the call times out, fails, or returns malformed
JSON, this module raises and the caller (executor.py) falls back to the
verified, already-tested rule-based parser. The rule-based parser remains
the default; this only activates if USE_LLM_PARSER=true and GROQ_API_KEY
is set.

Why add this now: allowed by the hackathon rules from day one ("free LLM
APIs ... may be used"), and safe to add this late only because it can never
break the existing, verified pipeline -- it's strictly additive.
"""

import os
import json
import requests

from intent_parser import QueryIntent

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"  # fast, free-tier model

SYSTEM_PROMPT = """You are an intent parser for an AML (anti-money-laundering) detection agent.
Given a natural language query, extract structured fields and respond with ONLY valid JSON,
no markdown, no explanation, matching exactly this schema:

{
  "query_type": one of ["single_entity", "structuring", "aggregation", "broad_analysis", "ml_anomaly", "unrelated"],
  "customer_id": string or null,
  "date_range_days": integer or null,
  "transaction_type": one of ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"] or null,
  "amount_threshold": number or null,
  "min_transaction_count": integer or null
}

Rules for classification:
- "single_entity": query asks about a specific customer/account ID (e.g. "Is customer ID X suspicious?")
- "structuring": query mentions structuring, smurfing, splitting, or layering patterns
- "aggregation": query asks "which customers", "how many", or gives a count/threshold condition
- "ml_anomaly": query explicitly asks for anomalies/patterns the rules might miss
- "broad_analysis": general/default for genuinely on-topic requests -- "analyse the dataset",
  no specific entity or pattern, but still about transactions/fraud/suspicious activity
- "unrelated": query has nothing to do with financial transactions, fraud, or suspicious
  activity at all (e.g. greetings, small talk, unrelated topics like weather or recipes).
  Do NOT default ambiguous or off-topic input to "broad_analysis" -- use "unrelated" instead.

Extract date ranges like "last 30 days" as date_range_days=30.
Extract amounts like "$10,000" as amount_threshold=10000.
Extract counts like "10+ transactions" as min_transaction_count=10.
Extract customer IDs like "C67886069" or "4521" as customer_id (as a string).
"""


def parse_query_llm(query: str, timeout: int = 8) -> QueryIntent:
    """
    Calls the LLM to parse intent. Raises on any failure (missing key,
    network error, bad JSON, unexpected schema) -- the caller is expected
    to catch this and fall back to the rule-based parser.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set -- LLM parsing unavailable")

    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)  # raises if the model didn't return valid JSON

    valid_types = {"single_entity", "structuring", "aggregation", "broad_analysis", "ml_anomaly", "unrelated"}
    if parsed.get("query_type") not in valid_types:
        raise ValueError(f"LLM returned unrecognized query_type: {parsed.get('query_type')}")

    tags = [f"[LLM-parsed] query_type={parsed['query_type']}"]

    return QueryIntent(
        raw_query=query,
        query_type=parsed["query_type"],
        customer_id=parsed.get("customer_id"),
        date_range_days=parsed.get("date_range_days"),
        transaction_type=parsed.get("transaction_type"),
        amount_threshold=parsed.get("amount_threshold"),
        min_transaction_count=parsed.get("min_transaction_count"),
        needs_full_eda=(parsed["query_type"] == "broad_analysis"),
        needs_ml_detection=(parsed["query_type"] not in ("aggregation", "single_entity")),
        tags=tags,
    )
