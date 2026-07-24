"""
planner.py

Takes a QueryIntent (from intent_parser.py) and builds a dynamic execution plan:
a list of tool names to call, in order, plus the arguments each one needs.

This is a deterministic if/else router - which is explicitly allowed by the
hackathon rules: "A deterministic orchestrated pipeline is acceptable if it
behaves like an agent and is clearly query-driven."

The key requirement it satisfies: NOT every query calls every tool.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from intent_parser import QueryIntent


@dataclass
class ExecutionStep:
    tool: str                      # name of the tool to call
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""                # human-readable justification, shown in the output


@dataclass
class ExecutionPlan:
    query: str
    query_type: str
    steps: List[ExecutionStep] = field(default_factory=list)
    decision_trail: List[str] = field(default_factory=list)  # mirrors intent.tags + planning decisions


def build_plan(intent: QueryIntent) -> ExecutionPlan:
    """
    Main entry point. Converts a QueryIntent into an ordered list of tool
    calls. This is the part that proves the agent isn't a fixed pipeline -
    the steps list is genuinely different per query type.
    """
    plan = ExecutionPlan(
        query=intent.raw_query,
        query_type=intent.query_type,
        decision_trail=list(intent.tags),  # start with the parser's reasoning
    )

    # ---------------------------------------------------------
    # CASE 1: Single-entity lookup
    # -> skip EDA, skip broad feature engineering, go straight to
    #    a targeted lookup + explanation for that one customer.
    # ---------------------------------------------------------
    if intent.query_type == "single_entity":
        plan.decision_trail.append(
            "Plan: single-entity lookup - skipping EDA and full-dataset feature engineering"
        )
        plan.steps = [
            ExecutionStep(
                tool="data_loader",
                args={"filter_customer_id": intent.customer_id},
                reason="Load only this customer's transaction history",
            ),
            ExecutionStep(
                tool="feature_engineering",
                args={"scope": "single_entity"},
                reason="Compute AML features for this customer only",
            ),
            ExecutionStep(
                tool="risk_classifier",
                args={"customer_id": intent.customer_id},
                reason="Compute or retrieve existing risk classification for this customer",
            ),
            ExecutionStep(
                tool="explanation_engine",
                args={"customer_id": intent.customer_id, "mode": "single_entity"},
                reason="Generate a plain-English explanation tied to this customer's flags",
            ),
        ]
        return plan

    # ---------------------------------------------------------
    # CASE 2: Structuring / specific AML pattern query
    # -> skip full EDA, run targeted feature engineering + detection only
    # ---------------------------------------------------------
    if intent.query_type == "structuring":
        plan.decision_trail.append(
            "Plan: structuring-focused query - skipping full EDA, running targeted detection only"
        )
        plan.steps = [
            ExecutionStep(
                tool="data_loader",
                args={"date_range_days": intent.date_range_days, "transaction_type": intent.transaction_type},
                reason="Load transactions filtered by date range/type (if specified)",
            ),
            ExecutionStep(
                tool="feature_engineering",
                args={"scope": "structuring", "features": ["nearThreshold10k", "senderVelocity", "senderRollingSum"]},
                reason="Compute only structuring-relevant features (near-threshold amounts, velocity)",
            ),
            ExecutionStep(
                tool="anomaly_detection",
                args={"mode": "hybrid", "pattern": "structuring"},
                reason="Run hybrid rule + ML detection scoped to structuring pattern",
            ),
            ExecutionStep(
                tool="risk_classifier",
                args={},
                reason="Classify flagged results into risk tiers",
            ),
            ExecutionStep(
                tool="explanation_engine",
                args={"mode": "structuring"},
                reason="Explain each flag in the context of structuring/smurfing behavior",
            ),
        ]
        return plan

    # ---------------------------------------------------------
    # CASE 3: Aggregation / threshold query
    # -> direct aggregation only, ML detection NOT required per spec
    # ---------------------------------------------------------
    if intent.query_type == "aggregation":
        plan.decision_trail.append(
            "Plan: aggregation/threshold query - running direct aggregation, skipping ML model entirely"
        )
        plan.steps = [
            ExecutionStep(
                tool="data_loader",
                args={"transaction_type": intent.transaction_type, "date_range_days": intent.date_range_days},
                reason="Load transactions matching any specified filters",
            ),
            ExecutionStep(
                tool="feature_engineering",
                args={"scope": "aggregation_only"},
                reason="Compute only the aggregate fields needed (counts, sums) - no ML features",
            ),
            ExecutionStep(
                tool="rule_engine",
                args={"amount_threshold": intent.amount_threshold},
                reason="Apply the threshold/count rule directly on the filtered data",
            ),
            ExecutionStep(
                tool="explanation_engine",
                args={"mode": "aggregation"},
                reason="Summarize the aggregation result in plain English",
            ),
        ]
        return plan

    # ---------------------------------------------------------
    # CASE 4: Broad analysis (default / fallback)
    # -> full pipeline: EDA -> features -> detection -> risk -> explanation
    # ---------------------------------------------------------
    plan.decision_trail.append(
        "Plan: broad analysis query - running full pipeline (EDA through explanation)"
    )
    plan.steps = [
        ExecutionStep(
            tool="data_loader",
            args={"date_range_days": intent.date_range_days},
            reason="Load full (or date-filtered) dataset",
        ),
        ExecutionStep(
            tool="eda_tool",
            args={},
            reason="Broad exploration requested - profile baseline behavior across the dataset",
        ),
        ExecutionStep(
            tool="feature_engineering",
            args={"scope": "full"},
            reason="Compute the full AML feature set",
        ),
        ExecutionStep(
            tool="anomaly_detection",
            args={"mode": "hybrid"},
            reason="Run full hybrid rule + ML anomaly detection",
        ),
        ExecutionStep(
            tool="risk_classifier",
            args={},
            reason="Classify all flagged results into risk tiers",
        ),
        ExecutionStep(
            tool="explanation_engine",
            args={"mode": "broad"},
            reason="Generate explanations for all flagged transactions/customers",
        ),
    ]
    return plan


if __name__ == "__main__":
    from intent_parser import parse_query

    test_queries = [
        "Find structuring patterns in the last 30 days",
        "Which customers made 10+ transactions under $10,000?",
        "Is customer ID 4521 suspicious?",
        "Analyse this dataset for suspicious activity",
    ]

    for q in test_queries:
        intent = parse_query(q)
        plan = build_plan(intent)
        print(f"\n{'='*70}\nQuery: {q}")
        print(f"Query type: {plan.query_type}")
        print("Decision trail:")
        for t in plan.decision_trail:
            print(f"  - {t}")
        print("Execution steps:")
        for i, step in enumerate(plan.steps, 1):
            print(f"  {i}. {step.tool}  (args={step.args})")
            print(f"     reason: {step.reason}")