# 🕵️ SentinelAML — Adaptive AI Agent for Suspicious Activity Detection

> An agentic AML (Anti-Money Laundering) system that dynamically plans its own investigation steps based on natural language queries — instead of running a fixed, one-size-fits-all pipeline.

---

## 📌 Problem Statement

Financial institutions are legally required to detect money laundering, but traditional rule-based AML systems generate excessive false positives (e.g., flagging every transaction over $10,000), overwhelming compliance teams while sophisticated techniques like **structuring**, **smurfing**, and **layering** slip through undetected.

**SentinelAML** solves this by acting as an autonomous compliance analyst: it reads a natural language query, decides _which tools it actually needs_ to answer it, runs only those, and explains its reasoning in plain English — reducing noise and giving compliance teams answers instead of alert floods.

This directly targets the same effectiveness/efficiency/explainability goals real-world AML platforms (e.g. Hawk AI, SymphonyAI) are built around, at hackathon scale.

---

## 🎯 What Makes This Different

Most AML tools — and most hackathon submissions — run the **same fixed pipeline** every time: load data → EDA → detect → score → done.

SentinelAML instead **routes each query differently**:

| Query                                                    | What the agent actually does                                                                                     |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `"Analyse this dataset for suspicious activity"`         | Runs the **full pipeline** — EDA → features → the validated full-balance-drain rule (100% precision / 97.5% recall) |
| `"Show me anomalous transactions the rules might miss"`  | Runs the **Isolation Forest ML model**, keeping only flags the rule engine does NOT already catch — the model's incremental value over rules |
| `"Is customer ID C67886069 suspicious?"`                 | Performs single-entity lookup → computes/explains risk for **just that customer**                                |
| `"Which customers made 10+ transactions under $10,000?"` | Runs direct aggregation + threshold rule → **skips ML model entirely**                                           |
| `"Find structuring patterns in the last 30 days"`        | Applies date filter → runs only the true structuring rule (near-$10k + repeat sender) → **skips full EDA and ML** |

The agent shows its own reasoning trail, so a reviewer can see _what it decided to do, and why_ — not just a final number.

> **Detection method per query type, and why:** broad-analysis and structuring queries use **rule-based detection only** — each has a validated rule signature (full-balance-drain: 100% precision / 97.5% recall; structuring: near-threshold amount + repeat-sender velocity) that's cheap, explainable, and outperforms adding the ML model or a weaker balance-inconsistency rule, both measured to collectively drop precision to ~1.7% for only a marginal recall gain. The **ML-anomaly query is where the Isolation Forest earns its place** — reserved for exactly the case rules can't cover: catching subtler, unknown patterns without a known signature. This is a deliberate precision/recall tradeoff per query type, not a blanket "always run everything" pipeline.

> **Note on the aggregation query:** PaySim customer IDs are largely single-use (max 3 transactions per sender), so multi-transaction structuring patterns don't naturally occur in this dataset. Our aggregation tool correctly returns 0 for this pattern — verified against raw data distribution.

> **Note on structuring detection scope:** the structuring query runs the true structuring rule only (near-$10,000 amount + repeat-sender velocity) — it excludes both unrelated balance-inconsistency flags and the general ML anomaly model, so every result it returns is honestly labeled as structuring rather than a generic anomaly score. Because repeat senders are rare in this dataset (see above), this query will usually return few or no results — that's expected behavior, not a bug.

---

## 🧠 Agent Architecture

```
                     ┌─────────────────────┐
   User Query   ───▶ │   Intent Parser      │
 (natural language)  │ (extracts intent,    │
                     │ filters, entities,   │
                     │ target AML pattern)  │
                     └──────────┬───────────┘
                                ▼
                     ┌─────────────────────┐
                     │      Planner        │
                     │ decides which tools │
                     │ to call, in what     │
                     │ order, on what data  │
                     └──────────┬───────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │              Tool Layer (called selectively)    │
        │                                                 │
        │  1. EDA Tool           → profiling & baselines   │
        │  2. Feature Engineering → velocity, rolling sums,│
        │                           amount deviation, etc. │
        │  3. Anomaly Detection   → rule-based / ML hybrid │
        │  4. Risk Classifier     → low / medium / high    │
        │  5. Explanation Engine  → plain-English reasons  │
        └──────────┬──────────────────────────────────────┘
                    ▼
        ┌─────────────────────────────────────┐
        │     Structured Agent Output          │
        │  • Execution summary (what/why)      │
        │  • Flagged transactions/customers    │
        │  • Risk level per flag                │
        │  • Explanation tied to query intent  │
        │  • Suggested SAR report draft         │
        │  • Escalation action (monitor/       │
        │    review/report)                     │
        └─────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer               | Tool                                                   |
| ------------------- | ------------------------------------------------------ |
| Agent orchestration | Python + LangGraph _(or custom router — see Notes)_    |
| Intent parsing      | Free-tier LLM API _(disclosed below)_                  |
| Feature engineering | pandas, numpy                                          |
| Anomaly detection   | Isolation Forest (ML) + rule-based thresholds (hybrid) |
| Visualization       | matplotlib / plotly                                    |
| Interface           | Streamlit chat UI _(or CLI, depending on time)_        |
| Data                | Public synthetic dataset (see below)                   |

> ⚠️ Replace the placeholders above with what you actually build — keep this table accurate to your final implementation.

---

## 📊 Dataset

- **Source:** [PaySim — Synthetic Financial Dataset for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) _(Kaggle, public)_
- **Why:** Simulates mobile money transactions with labeled fraudulent activity, closely mirroring structuring/smurfing-style behavior.
- **License:** Open/public dataset, no proprietary or confidential data used.
- **Schema summary:** step (time), type (transaction type), amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud.
- Any synthetic data generated beyond this dataset (e.g. to simulate structuring patterns) is documented in `/data/synthetic_notes.md` with schema, assumptions, and generation logic.

---

## 🚀 Setup & Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd sentinel-aml

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your free-tier LLM API key (if used)
cp .env.example .env
# then fill in your API key in .env

# 5. Run the app
streamlit run app.py
```

---

## 💬 Usage

Once running, type a natural language query into the interface. Recommended demo order:

```
> "Analyse this dataset for suspicious activity"          # rule-based full pipeline -- real flagged transactions, risk tiers, SAR drafts
> "Show me anomalous transactions the rules might miss"   # ML model's incremental value over the validated rule
> "Is customer ID C1889568678 suspicious?"                # single-entity lookup, positive hit -- HIGH risk, full-balance-drain rule
> "Is customer ID C67886069 suspicious?"                  # single-entity lookup, negative -- clean history, no false positive
> "Which customers made 10+ transactions under $10,000?"  # direct aggregation -- skips ML entirely
> "Find structuring patterns in the last 30 days"         # targeted structuring rule only -- see note below
```

The agent responds with:

1. **Execution summary** — what it detected in your query and which tools it chose to invoke
2. **Flagged results** — transactions/customers with risk level
3. **Explanation** — plain-English reason for each flag
4. **Suggested SAR draft** — auto-generated summary formatted like a Suspicious Activity Report
5. **Recommended action** — monitor / review / report

> **Why lead with the broad-analysis query:** it's the one with the validated rule-based pipeline and visible output — real flagged transactions, risk levels, and SAR drafts (8,008 flags at 100% precision / 97.5% recall against ground-truth fraud labels). Follow it with the ML-anomaly query to show the model's incremental value — what it catches beyond the rule. The single-entity, aggregation, and structuring queries are just as correct, but two of them return **0 results on this dataset by design** (see notes above) — that's the agent refusing to manufacture false positives, not a gap. Lead the demo with results, then use the "honest zero" queries to show the explainability story: an agent that tells you a pattern doesn't exist instead of dressing up unrelated anomalies to produce a number.

---

## 📝 Suggested SAR Report Draft (Feature Highlight)

Beyond just flagging risk, SentinelAML drafts a short, human-readable **Suspicious Activity Report (SAR) summary** for each high-risk case — reframing the explanation engine's output into a compliance-ready format, e.g.:

> _"Customer X moved funds via 5 transactions of $9,200–$9,800 within a 48-hour window, each just under the $10,000 reporting threshold — consistent with a structuring pattern. Recommended action: escalate for review."_

This mirrors how modern AML platforms use generative AI to automate case triage and draft SARs, at a scope appropriate for a 48-hour build.

---

## 🔮 Future Work (Not Built — Scoped Out Intentionally)

- **Graph-based entity-link analysis**: extending detection to uncover hidden relationships across accounts, shared devices, or shared emails to catch coordinated fraud rings (used by modern platforms like Hawk AI). Not attempted here due to time constraints — flagged as a clear next step rather than a half-built feature.
- **Real-time streaming detection**: current scope is batch analysis on a sample dataset, per hackathon guidance.
- **Full case management workflow**: multi-analyst review, audit trails, and regulator-facing reporting.

---

## 🤖 Disclosed AI/Tool Usage

_(Fill in exactly what you used — required for submission)_

- LLM API used: `[e.g. free-tier Groq / OpenRouter / Gemini API]`
- Agentic coding assistance used: `[e.g. Claude Code / GitHub Copilot]`
- Libraries: pandas, scikit-learn, LangGraph, Streamlit _(update to match final build)_

---

## 👥 Team

`[Your names here]`

---

## ⚠️ Notes

- No proprietary, confidential, or copyrighted data used — all data is public/synthetic, sourced and cited above.
- This is a hackathon prototype, not a production-grade system, per scope guidance.
