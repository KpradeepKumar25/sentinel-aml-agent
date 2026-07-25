"""
app.py

Streamlit UI for SentinelAML. Wraps executor.run_agent() in a visual
interface: type a natural-language query, watch the agent's decision
trail render as a pipeline, and inspect flagged results with risk
badges and SAR drafts.

Run from src/agent/:
    streamlit run app.py
"""

import sys
import os
import time
import streamlit as st

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "tools"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "utils"))

from executor import run_agent


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SentinelAML",
    page_icon="\U0001F6F0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #0A0E14;
    --bg-panel: #121820;
    --bg-panel-alt: #161D27;
    --border-subtle: #22303C;
    --accent: #2DD4BF;
    --accent-dim: #1A8F82;
    --text-primary: #E6EDF3;
    --text-secondary: #8B98A5;
    --risk-high: #EF4444;
    --risk-medium: #F59E0B;
    --risk-low: #3B82F6;
    --risk-none: #4B5563;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 0%, #0D1420 0%, #0A0E14 55%);
    color: var(--text-primary);
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-panel);
    border-right: 1px solid var(--border-subtle);
}

/* Hero header */
.sentinel-hero {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 6px 0 18px 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 24px;
}
.sentinel-eye {
    width: 42px; height: 42px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 24px rgba(45, 212, 191, 0.35);
}
.sentinel-title {
    font-size: 26px; font-weight: 700; letter-spacing: -0.5px;
    color: var(--text-primary); margin: 0;
}
.sentinel-subtitle {
    font-size: 13px; color: var(--text-secondary); margin: 0;
    letter-spacing: 0.3px; text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 18px;
    height: 100%;
}
.metric-label {
    font-size: 11px; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px; font-weight: 600; color: var(--accent);
}

/* Pipeline steps */
.pipeline-wrap {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 20px;
}
.pipeline-title {
    font-size: 12px; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px;
}
.pipeline-step {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 10px 0;
    border-left: 2px solid var(--accent-dim);
    padding-left: 16px;
    margin-left: 10px;
    position: relative;
}
.pipeline-step::before {
    content: '';
    position: absolute;
    left: -7px; top: 16px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px rgba(45, 212, 191, 0.6);
}
.pipeline-step-name {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600; color: var(--text-primary); font-size: 14px;
}
.pipeline-step-reason {
    color: var(--text-secondary); font-size: 13px; margin-top: 2px;
}

.skipped-tools {
    margin-top: 12px; padding-top: 12px;
    border-top: 1px dashed var(--border-subtle);
    font-size: 12px; color: var(--text-secondary);
}
.skipped-tag {
    display: inline-block;
    background: rgba(75, 85, 99, 0.25);
    color: var(--text-secondary);
    padding: 3px 10px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; margin: 3px 4px 0 0;
    text-decoration: line-through;
}

/* Risk badges */
.risk-badge {
    display: inline-block;
    padding: 3px 12px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
}
.risk-HIGH { background: rgba(239,68,68,0.15); color: var(--risk-high); border: 1px solid rgba(239,68,68,0.4); }
.risk-MEDIUM { background: rgba(245,158,11,0.15); color: var(--risk-medium); border: 1px solid rgba(245,158,11,0.4); }
.risk-LOW { background: rgba(59,130,246,0.15); color: var(--risk-low); border: 1px solid rgba(59,130,246,0.4); }
.risk-NONE { background: rgba(75,85,99,0.15); color: var(--risk-none); border: 1px solid rgba(75,85,99,0.4); }
.risk-UNKNOWN { background: rgba(75,85,99,0.15); color: var(--risk-none); border: 1px solid rgba(75,85,99,0.4); }

/* Result card */
.result-card {
    background: var(--bg-panel-alt);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.result-flow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px; color: var(--text-primary);
    margin-bottom: 6px;
}
.result-explanation {
    font-size: 13px; color: var(--text-secondary);
    margin-bottom: 10px; line-height: 1.5;
}
.sar-box {
    background: #0D1117;
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #B8C4CE;
    white-space: pre-wrap;
    line-height: 1.6;
}

.stButton>button {
    background: var(--accent);
    color: #06201C;
    border: none;
    font-weight: 600;
    border-radius: 8px;
}
.stButton>button:hover {
    background: var(--accent-dim);
    color: white;
}

.example-query-btn button {
    background: var(--bg-panel-alt) !important;
    color: var(--text-secondary) !important;
    font-weight: 400 !important;
    text-align: left !important;
    font-size: 12px !important;
    border: 1px solid var(--border-subtle) !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="sentinel-hero">
    <div class="sentinel-eye">&#128065;</div>
    <div>
        <p class="sentinel-title">SentinelAML</p>
        <p class="sentinel-subtitle">Adaptive AI Agent &middot; Suspicious Activity Detection</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — query input + examples
# ============================================================
EXAMPLE_QUERIES = [
    "Analyse this dataset for suspicious activity",
    "Show anomalous transactions the rules might miss",
    "Find structuring patterns in the last 30 days",
    "Which customers made 10+ transactions under $10,000?",
    "Is customer ID C67886069 suspicious?",
]

with st.sidebar:
    st.markdown("#### Ask the agent")
    st.caption("Type a query, or pick a verified example below.")

    if "query_text" not in st.session_state:
        st.session_state.query_text = EXAMPLE_QUERIES[0]

    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.query_text = q

    st.markdown("---")
    st.markdown("#### Architecture")
    st.caption(
        "Intent Parser -> Planner -> Executor\n\n"
        "The planner invokes only the tools each query type needs — "
        "EDA, feature engineering, rule-based detection, ML anomaly "
        "detection, risk classification, and explanation, selectively."
    )

# ============================================================
# MAIN — query box + run
# ============================================================
query = st.text_input(
    "Query",
    value=st.session_state.query_text,
    label_visibility="collapsed",
    placeholder="e.g. Is customer ID C67886069 suspicious?",
)
run_clicked = st.button("Run agent", type="primary")

if run_clicked and query.strip():
    with st.spinner("Agent is parsing intent and building an execution plan..."):
        start = time.time()
        try:
            result = run_agent(query)
        except Exception as e:
            st.error(f"Agent execution failed: {e}")
            st.stop()
        elapsed = time.time() - start

    # --- Top metrics row ---
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Query type", result["query_type"].replace("_", " ").title()),
        ("Rows analyzed", f"{result['total_rows_analyzed']:,}"),
        ("Flagged", f"{result['flagged_count']:,}"),
        ("Runtime", f"{elapsed:.1f}s"),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --- Decision trail / pipeline ---
    all_tools = ["data_loader", "eda_tool", "feature_engineering", "anomaly_detection",
                 "rule_engine", "risk_classifier", "explanation_engine"]
    invoked = result["tools_invoked"]
    skipped = [t for t in all_tools if t not in invoked]

    steps_html = ""
    for i, (tool, reason) in enumerate(zip(invoked, result["decision_trail"] + [""] * len(invoked))):
        steps_html += f"""
        <div class="pipeline-step">
            <div>
                <div class="pipeline-step-name">{i+1}. {tool}</div>
            </div>
        </div>
        """

    trail_html = "".join(
        f'<div class="pipeline-step-reason" style="margin-left: 26px; margin-bottom: 8px;">&rarr; {t}</div>'
        for t in result["decision_trail"]
    )

    skipped_html = "".join(f'<span class="skipped-tag">{t}</span>' for t in skipped)

    st.markdown(f"""
    <div class="pipeline-wrap">
        <div class="pipeline-title">Agent Decision Trail</div>
        {trail_html}
        <div style="margin-top: 14px; padding-top: 14px; border-top: 1px dashed var(--border-subtle);">
            <div class="pipeline-title" style="margin-bottom: 8px;">Tools Invoked</div>
            {"".join(f'<span class="risk-badge risk-LOW" style="margin-right:6px; margin-bottom:6px; display:inline-block;">{t}</span>' for t in invoked)}
        </div>
        <div class="skipped-tools">
            Skipped (not needed for this query): {skipped_html if skipped_html else "none"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- EDA summary, if present ---
    if result.get("eda_summary"):
        sampling_note = result["eda_summary"].get("sampling_note")
        if sampling_note:
            st.warning(f"**Sampling notice:** {sampling_note}", icon="⚠️")
        with st.expander("EDA Summary", expanded=False):
            st.json(result["eda_summary"])

    # --- Flagged results ---
    st.markdown("#### Flagged Results")
    flagged = result.get("flagged_results", [])

    if not flagged:
        st.info(
            "No transactions flagged for this query. This can be a correct, "
            "meaningful result — e.g. a pattern that structurally doesn't "
            "occur in the filtered data — not necessarily an empty/broken output."
        )
    else:
        for r in flagged[:15]:
            risk = r.get("risk_level", "UNKNOWN")
            st.markdown(f"""
            <div class="result-card">
                <div class="result-flow">{r.get('sender')} &rarr; {r.get('receiver')}
                    &nbsp;&middot;&nbsp; ${r.get('amount', 0):,.2f}
                    &nbsp;<span class="risk-badge risk-{risk}">{risk}</span>
                </div>
                <div class="result-explanation">{r.get('explanation', '')}</div>
                <div class="sar-box">{r.get('suggested_sar', '')}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(flagged) > 15:
            st.caption(f"Showing 15 of {len(flagged)} flagged results.")

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: var(--text-secondary);">
        <div style="font-size: 40px; margin-bottom: 12px;">&#128269;</div>
        <div style="font-size: 15px;">Enter a query and click <b>Run agent</b> to see the adaptive execution plan and results.</div>
    </div>
    """, unsafe_allow_html=True)