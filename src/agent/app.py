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
import pandas as pd
import altair as alt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "tools"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "utils"))

# On Streamlit Community Cloud there is no .env file -- secrets are set via
# the app's Settings > Secrets panel instead and surface through st.secrets.
# executor.py only reads os.environ (via python-dotenv locally), so mirror
# any matching secrets into the process env before importing it. This is a
# no-op locally (no secrets.toml) and a no-op on the cloud unless the two
# optional keys below were actually configured.
try:
    for _key in ("GROQ_API_KEY", "USE_LLM_PARSER"):
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

from executor import run_agent


# ============================================================
# CHART HELPERS
# ============================================================
CHART_AXIS_COLOR = "#8B98A5"
CHART_GRID_COLOR = "#22303C"
RISK_ORDER = ["HIGH", "MEDIUM", "LOW", "NONE", "UNKNOWN"]
# Reuses the exact status colors already defined for the risk badges above
# (.risk-HIGH / .risk-MEDIUM / .risk-LOW / .risk-NONE) so the chart and the
# badges never disagree about what a risk tier looks like.
RISK_COLORS = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#3B82F6", "NONE": "#4B5563", "UNKNOWN": "#4B5563"}


def _bar_chart(df, x_field, y_field, x_title, y_title, sort_order, color=None,
                color_field=None, color_scale=None, height=180, log_scale=False):
    """Thin rounded bars on a transparent background, styled to match the
    app's dark theme. Tooltip-on-hover comes for free via Altair/Vega-Lite.

    log_scale=True switches the y-axis to symlog (handles zero, unlike a
    plain log scale) -- use it when categories can differ by orders of
    magnitude (e.g. 8,008 HIGH vs 12 MEDIUM), where a linear scale makes
    the smaller bar's real, non-zero count visually indistinguishable from
    zero. Count labels are always drawn above each bar regardless of scale,
    so the exact number is legible even when the bar itself is short.
    """
    y_scale = alt.Scale(type="symlog") if log_scale else alt.Undefined
    x_enc = alt.X(f"{x_field}:N", title=x_title, sort=sort_order,
                  axis=alt.Axis(labelColor=CHART_AXIS_COLOR, titleColor=CHART_AXIS_COLOR,
                                 domain=False, ticks=False, labelAngle=0))
    y_enc = alt.Y(f"{y_field}:Q", title=y_title, scale=y_scale,
                  axis=alt.Axis(labelColor=CHART_AXIS_COLOR, titleColor=CHART_AXIS_COLOR,
                                 gridColor=CHART_GRID_COLOR, domain=False, ticks=False))
    tooltip = [alt.Tooltip(f"{x_field}:N", title=x_title), alt.Tooltip(f"{y_field}:Q", title=y_title, format=",")]

    if color_field and color_scale:
        color_enc = alt.Color(f"{color_field}:N", scale=color_scale, legend=None)
    else:
        color_enc = alt.value(color or "#2DD4BF")

    base = alt.Chart(df)
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=34).encode(
        x=x_enc, y=y_enc, color=color_enc, tooltip=tooltip,
    )
    labels = base.mark_text(dy=-8, baseline="bottom", color=CHART_AXIS_COLOR, fontSize=11).encode(
        x=x_enc, y=y_enc, text=alt.Text(f"{y_field}:Q", format=","),
    )

    return (
        alt.layer(bars, labels)
        .properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
    )


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

/* Entrance animation -- subtle, one-shot, not a gimmick */
@keyframes sentinelFadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Hero header */
.sentinel-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0 18px 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 24px;
    animation: sentinelFadeUp 0.5s ease-out;
}
.sentinel-hero {
    display: flex;
    align-items: center;
    gap: 14px;
}
.sentinel-eye {
    width: 42px; height: 42px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 24px rgba(45, 212, 191, 0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.sentinel-eye:hover {
    transform: scale(1.08) rotate(-4deg);
    box-shadow: 0 0 32px rgba(45, 212, 191, 0.55);
}
.sentinel-title {
    font-size: 26px; font-weight: 700; letter-spacing: -0.5px;
    color: var(--text-primary); margin: 0;
}
.sentinel-subtitle {
    font-size: 13px; color: var(--text-secondary); margin: 0;
    letter-spacing: 0.3px; text-transform: uppercase;
}
.team-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 600; color: var(--accent);
    background: rgba(45, 212, 191, 0.08);
    border: 1px solid rgba(45, 212, 191, 0.3);
    padding: 6px 14px; border-radius: 20px;
    letter-spacing: 0.3px;
    transition: background 0.2s ease, box-shadow 0.2s ease;
}
.team-badge:hover {
    background: rgba(45, 212, 191, 0.16);
    box-shadow: 0 0 16px rgba(45, 212, 191, 0.25);
}

/* Metric cards */
.metric-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 18px;
    height: 100%;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    animation: sentinelFadeUp 0.5s ease-out;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent-dim);
    box-shadow: 0 6px 20px rgba(45, 212, 191, 0.12);
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
    transition: padding-left 0.2s ease;
}
.pipeline-step:hover {
    padding-left: 20px;
}
.pipeline-step:hover .pipeline-step-name {
    color: var(--accent);
}
.pipeline-step::before {
    content: '';
    position: absolute;
    left: -7px; top: 16px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px rgba(45, 212, 191, 0.6);
    transition: box-shadow 0.2s ease;
}
.pipeline-step:hover::before {
    box-shadow: 0 0 14px rgba(45, 212, 191, 0.9);
}
.pipeline-step-name {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600; color: var(--text-primary); font-size: 14px;
    transition: color 0.2s ease;
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
    transition: filter 0.2s ease, transform 0.2s ease;
}
.risk-badge:hover {
    filter: brightness(1.25);
    transform: scale(1.06);
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
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    animation: sentinelFadeUp 0.4s ease-out;
}
.result-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-dim);
    box-shadow: 0 6px 20px rgba(45, 212, 191, 0.1);
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
    transition: background 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
}
.stButton>button:hover {
    background: var(--accent-dim);
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(45, 212, 191, 0.35);
}
.stButton>button:active {
    transform: translateY(0px);
}

.example-query-btn button {
    background: var(--bg-panel-alt) !important;
    color: var(--text-secondary) !important;
    font-weight: 400 !important;
    text-align: left !important;
    font-size: 12px !important;
    border: 1px solid var(--border-subtle) !important;
}

/* Focus glow on inputs -- makes keyboard/click focus feel alive */
.stTextInput input:focus, div[data-baseweb="select"]:focus-within > div {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

/* Footer */
.app-footer {
    margin-top: 48px;
    padding: 20px 0 12px 0;
    border-top: 1px solid var(--border-subtle);
    text-align: center;
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.8;
}
.app-footer b { color: var(--text-primary); }
.app-footer a { color: var(--accent); text-decoration: none; transition: color 0.2s ease; }
.app-footer a:hover { color: var(--accent-dim); text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="sentinel-topbar">
    <div class="sentinel-hero">
        <div class="sentinel-eye">&#128065;</div>
        <div>
            <p class="sentinel-title">SentinelAML</p>
            <p class="sentinel-subtitle">Adaptive AI Agent &middot; Suspicious Activity Detection</p>
        </div>
    </div>
    <div class="team-badge">Built by White Hats</div>
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
    "Is customer ID C999000001 suspicious?",
    "Is customer ID C67886069 suspicious?",
]

EXAMPLE_PLACEHOLDER = "-- type to search examples --"


def _apply_example_pick():
    choice = st.session_state.example_picker
    if choice != EXAMPLE_PLACEHOLDER:
        st.session_state.query_text = choice


if "query_text" not in st.session_state:
    st.session_state.query_text = EXAMPLE_QUERIES[0]

with st.sidebar:
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
# The example picker lives here (not just the sidebar) because the sidebar
# can be collapsed, which would otherwise hide the only 6 queries verified
# to parse correctly -- free text beyond these is best-effort.
st.caption("Pick one of the 6 verified examples, or type your own query below.")
st.selectbox(
    "Examples",
    options=[EXAMPLE_PLACEHOLDER] + EXAMPLE_QUERIES,
    key="example_picker",
    on_change=_apply_example_pick,
    label_visibility="collapsed",
)

query = st.text_input(
    "Query",
    value=st.session_state.query_text,
    label_visibility="collapsed",
    placeholder="e.g. Is customer ID C67886069 suspicious?",
)
run_clicked = st.button("Run agent", type="primary")

# Compute + cache into session_state on click. Reading the cached result
# below (rather than gating everything on `run_clicked`) is what lets other
# widgets on the page -- pagination, expanders -- rerun the script without
# wiping the display: `run_clicked` is only True on the exact rerun the
# button itself triggered, and would be False again on every subsequent
# rerun (e.g. clicking "Next" on the results pager).
if run_clicked and query.strip():
    with st.spinner("Agent is parsing intent and building an execution plan..."):
        start = time.time()
        try:
            computed_result = run_agent(query)
        except Exception as e:
            st.error(f"Agent execution failed: {e}")
            st.stop()
        computed_elapsed = time.time() - start
    st.session_state.last_result = computed_result
    st.session_state.last_elapsed = computed_elapsed

if "last_result" in st.session_state:
    result = st.session_state.last_result
    elapsed = st.session_state.last_elapsed

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
        eda = result["eda_summary"]
        sampling_note = eda.get("sampling_note")
        if sampling_note:
            st.warning(f"**Sampling notice:** {sampling_note}", icon="⚠️")

        type_counts = eda.get("transaction_type_counts")
        if type_counts:
            st.markdown("###### Transaction Type Breakdown")
            t_order = sorted(type_counts, key=lambda k: -type_counts[k])
            t_df = pd.DataFrame({"type": t_order, "count": [type_counts[t] for t in t_order]})
            st.altair_chart(
                _bar_chart(t_df, "type", "count", "Type", "Transactions", t_order),
                use_container_width=True,
            )

        with st.expander("Full EDA Summary (JSON)", expanded=False):
            st.json(eda)

    # --- Flagged results ---
    st.markdown("#### Flagged Results")
    flagged = result.get("flagged_results", [])

    # Risk breakdown is computed backend-side across the FULL flagged set
    # (before the 50-row display cap), so it reflects the true distribution
    # even when only a page of results is shown below.
    risk_breakdown = result.get("risk_level_breakdown", {})
    if risk_breakdown:
        r_order = [r for r in RISK_ORDER if r in risk_breakdown]
        r_order += [r for r in risk_breakdown if r not in r_order]
        r_df = pd.DataFrame({"risk_level": r_order, "count": [risk_breakdown[r] for r in r_order]})
        st.markdown("###### Risk Level Breakdown (all flagged, not just the page shown below)")
        st.altair_chart(
            _bar_chart(
                r_df, "risk_level", "count", "Risk level", "Count", r_order,
                color_field="risk_level",
                color_scale=alt.Scale(domain=list(RISK_COLORS.keys()), range=list(RISK_COLORS.values())),
                log_scale=True,
            ),
            use_container_width=True,
        )

    if flagged:
        with st.expander("View all shown results as a table", expanded=False):
            table_df = pd.DataFrame([{
                "Sender": r.get("sender"),
                "Receiver": r.get("receiver"),
                "Amount": r.get("amount", 0),
                "Risk": r.get("risk_level", "UNKNOWN"),
                "Action": r.get("recommended_action", ""),
                "Explanation": r.get("explanation", ""),
            } for r in flagged])
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                column_config={"Amount": st.column_config.NumberColumn(format="$%,.2f")},
            )

    if not flagged:
        st.info(
            "No transactions flagged for this query. This can be a correct, "
            "meaningful result — e.g. a pattern that structurally doesn't "
            "occur in the filtered data — not necessarily an empty/broken output."
        )
    else:
        PAGE_SIZE = 15
        returned_total = len(flagged)          # rows actually returned by the agent (capped at 50)
        true_total = result.get("flagged_count", returned_total)  # true count before that cap
        total_pages = max(1, -(-returned_total // PAGE_SIZE))  # ceil division

        # Reset to page 1 whenever a different query is run
        if st.session_state.get("last_query_run") != query:
            st.session_state.results_page = 1
            st.session_state.last_query_run = query

        page = min(max(st.session_state.get("results_page", 1), 1), total_pages)
        start = (page - 1) * PAGE_SIZE
        end = start + PAGE_SIZE

        for r in flagged[start:end]:
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

        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("← Previous", disabled=(page <= 1), use_container_width=True):
                st.session_state.results_page = page - 1
                st.rerun()
        with nav2:
            caption = f"Page {page} of {total_pages} — showing {start + 1}-{min(end, returned_total)} of {returned_total}"
            if true_total > returned_total:
                caption += f" (top {returned_total} of {true_total:,} total flagged)"
            st.markdown(
                f"<div style='text-align:center; padding-top:8px; color:var(--text-secondary); font-size:13px;'>{caption}</div>",
                unsafe_allow_html=True,
            )
        with nav3:
            if st.button("Next →", disabled=(page >= total_pages), use_container_width=True):
                st.session_state.results_page = page + 1
                st.rerun()

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: var(--text-secondary);">
        <div style="font-size: 40px; margin-bottom: 12px;">&#128269;</div>
        <div style="font-size: 15px;">Enter a query and click <b>Run agent</b> to see the adaptive execution plan and results.</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="app-footer">
    &copy; 2026 SentinelAML &mdash; Built by <b>White Hats</b><br>
    K Pradeep Kumar (<a href="mailto:kumarkpradeep2005@gmail.com">kumarkpradeep2005@gmail.com</a>)
    &middot;
    P Vahran (<a href="mailto:vahranp@gmail.com">vahranp@gmail.com</a>)
</div>
""", unsafe_allow_html=True)