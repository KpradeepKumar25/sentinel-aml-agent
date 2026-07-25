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

Broad analysis correctly flags both drain-signature fraud and structuring patterns — including cases not labeled as fraud by the dataset itself, demonstrating detection beyond ground-truth labels.

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

> **Note on the aggregation query:** PaySim customer IDs are largely single-use (max 3 transactions per sender across the full 6.3M-row dataset), so multi-transaction structuring patterns don't naturally occur in this data — verified against raw data distribution. On unmodified PaySim data this query correctly returns 0. On the demo sample specifically, it returns **12** — the disclosed synthetic structuring block (sender `C999000001`, see Dataset section) added to demonstrate the pathway genuinely fires when the pattern is present.

> **Note on structuring detection scope:** the structuring query runs the true structuring rule only (near-$10,000 amount + repeat-sender velocity) — it excludes both unrelated balance-inconsistency flags and the general ML anomaly model, so every result it returns is honestly labeled as structuring rather than a generic anomaly score. On unmodified PaySim data this returns few or no results, since repeat senders are structurally rare (see above) — that's expected behavior, not a bug. The demo sample includes a disclosed synthetic structuring block for exactly this reason: to prove the rule works correctly when its target pattern actually exists, rather than asking a reviewer to trust an always-empty result.

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

| Layer               | Tool                                                                                          |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| Agent orchestration | Python — custom deterministic router (Intent Parser → Planner → Executor, no LangGraph)         |
| Intent parsing      | Deterministic keyword/regex-based parser by default; optional Groq LLM parser with automatic fallback (see Disclosed AI/Tool Usage) |
| Feature engineering | pandas, numpy                                                                                    |
| Anomaly detection   | Rule-based (validated: full-balance-drain rule, 100% precision / 97.5% recall) + Isolation Forest ML, applied selectively per query type — hybrid architecture, not always both at once |
| Visualization       | Altair (via Streamlit)                                                                           |
| Interface           | Streamlit web UI                                                                                 |
| Data                | PaySim (public, Kaggle) + a small disclosed synthetic block (see Dataset section)                |

---

## 📊 Dataset

- **Source:** [PaySim — Synthetic Financial Dataset for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) _(Kaggle, public)_
- **Why:** Simulates mobile money transactions with labeled fraudulent activity, closely mirroring structuring/smurfing-style behavior.
- **License:** Open/public dataset, no proprietary or confidential data used.
- **Schema summary:** step (time), type (transaction type), amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud.

> **Filter coverage — a dataset limitation, not a gap:** PaySim's schema has no `segment` or `country` fields, so the intent parser doesn't extract those filters — there's nothing in the data to filter on. Date range, transaction type, amount threshold, and entity ID filters are all supported and demonstrated across the example queries above.

> **Disclosed synthetic data — structuring demo block:** to demonstrate structuring detection — a pattern requiring a repeat sender, which doesn't naturally occur in PaySim's single-use customer IDs (verified: max 3 transactions per sender across the full 6.3M-row dataset) — we added 12 clearly-labeled synthetic transactions (sender `C999000001`) simulating classic structuring behavior (multiple transfers just under the $10,000 reporting threshold, spread across the last ~20 simulated days). Generation logic: [`add_synthetic_structuring_demo.py`](add_synthetic_structuring_demo.py). These synthetic rows are intentionally **NOT** labeled as fraud (`isFraud=0`) in the dataset, since structuring is designed to evade naive fraud detection — demonstrating that our rule catches this pattern independent of ground-truth labels, closer to how real-world detection has to work (no answer key in production). This is disclosed, rule-compliant use of synthetic data per the hackathon's own rules on synthetic data usage. Verified end-to-end: the structuring query flags all 12 synthetic rows at **MEDIUM** risk (not HIGH — see the note below on why the two rule types get different tiers), the aggregation query correctly groups them by sender, and the single-entity lookup for `C999000001` correctly isolates just these 12 transactions.

---

## 🚀 Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/KpradeepKumar25/sentinel-aml-agent.git
cd sentinel_aml_agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app (no API key needed -- intent parsing is a local
#    deterministic parser, not an LLM call; see Disclosed AI/Tool Usage)
cd src/agent
streamlit run app.py
```

---

## ☁️ Deployment (Streamlit Community Cloud)

The app is a standard Streamlit app and deploys as a free, public website with
no code changes beyond what's already in this repo:

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick this repo/branch, and set the main file path to
   `src/agent/app.py`.
4. *(Optional)* To enable the optional Groq LLM intent parser on the deployed
   app, open **Settings → Secrets** and paste the two values from
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) with
   your own key. Leave this step out entirely and the app runs exactly the
   same, using the default local regex parser.
5. Click **Deploy**. First boot installs `requirements.txt` and typically
   takes 2-3 minutes; the bundled `data/raw/paysim_sample.csv` and pretrained
   `src/models/*.pkl` are already committed, so no external downloads are
   needed at runtime.

`.streamlit/config.toml` pins the dark theme so the deployed app matches the
local one, and `runtime.txt` pins the Python version to match what the
bundled model was validated against.

Any other host that runs a standard Python web app (Render, Railway, Hugging
Face Spaces, Google Cloud Run) works too — the entry point is
`streamlit run src/agent/app.py`, the dependency list is `requirements.txt`,
and the two optional secrets are read from environment variables.

---

## 💬 Usage

Once running, type a natural language query into the interface. Recommended demo order:

```
> "Analyse this dataset for suspicious activity"          # rule-based full pipeline -- real flagged transactions, risk tiers, SAR drafts
> "Show me anomalous transactions the rules might miss"   # ML model's incremental value over the validated rule
> "Is customer ID C1889568678 suspicious?"                # single-entity lookup, positive hit -- HIGH risk, full-balance-drain rule
> "Is customer ID C67886069 suspicious?"                  # single-entity lookup, negative -- clean history, no false positive
> "Is customer ID C999000001 suspicious?"                 # single-entity lookup on the disclosed synthetic structuring sender -- 12/12 correctly flagged
> "Which customers made 10+ transactions under $10,000?"  # direct aggregation -- skips ML entirely, catches the synthetic block (12 rows)
> "Find structuring patterns in the last 30 days"         # targeted structuring rule only -- catches the synthetic block (12 rows), see notes below
```

The agent responds with:

1. **Execution summary** — what it detected in your query and which tools it chose to invoke
2. **Flagged results** — transactions/customers with risk level
3. **Explanation** — plain-English reason for each flag
4. **Suggested SAR draft** — auto-generated summary formatted like a Suspicious Activity Report
5. **Recommended action** — monitor / review / report

> **Why lead with the broad-analysis query:** it's the one with the validated rule-based pipeline and visible output — real flagged transactions, risk levels, and SAR drafts. On the demo sample it flags **8,020** transactions total: **8,008** via the full-balance-drain rule (100% precision / 97.5% recall against `isFraud`) plus **12** via the structuring rule — the same disclosed synthetic block (sender `C999000001`) also picked up here because `rule_based_flag()` checks the structuring condition for every query, not just ones that pass `pattern="structuring"`. That's a stronger explainability story, not a discrepancy: broad-analysis demonstrably catches both drain-signature fraud *and* structuring, including a case intentionally **not** labeled as fraud in `isFraud` (structuring is designed to evade naive fraud labels — see Dataset section). Report precision per rule, not as one blended number: the drain rule's 100%/97.5% holds against `isFraud` for its own 8,008; the structuring rule's 12/12 are validated against the known ground truth that we constructed them as structuring, not against `isFraud`, which was never meant to capture that pattern. On unmodified PaySim data (no synthetic block), broad-analysis returns exactly 8,008 and the aggregation/structuring queries correctly return 0 — that's the agent refusing to manufacture false positives on data that structurally can't contain the pattern, not a gap.

> **Risk tiers reflect rule confidence, not just "flagged or not":** the two rules feeding `rules_only` mode aren't equally validated, so they don't get the same tier. The drain rule (100% precision / 97.5% recall against `isFraud`) maps to **HIGH**. The structuring rule — only demonstrated on the disclosed synthetic block, never validated against real fraud labels — maps to **MEDIUM**, i.e. "flag for review" rather than "report." So the broad-analysis query's risk breakdown is genuinely mixed: **8,008 HIGH + 12 MEDIUM**, not one flat tier — the risk level is doing real work, not decoration.

> **The ML-anomaly query's risk tiers, explained:** for `"Show me anomalous transactions the rules might miss"`, tiers come from ranking the model's `ml_anomaly_score` *within the flagged set itself* (not against the whole dataset — almost anything already flagged clears that low bar, which is why an earlier version of this put ~86% of flags in one bucket). Checked against `isFraud`: the model's own top 10% scores **67.5%** precision (114 rows) → **HIGH**; the next band scores **34%** (229 rows) → **MEDIUM**; the rest scores **~0.6%** (1,039 rows) → **LOW**. A real, validated three-way split, not an even/decorative one — most of what Isolation Forest flags here is genuinely low-confidence, and the tiers say so honestly.

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

- **LLM API used (optional, off by default):** intent parsing defaults to a local,
  deterministic keyword/regex-based parser (`src/agent/intent_parser.py`) — no runtime
  LLM call is made unless explicitly enabled. Setting `USE_LLM_PARSER=true` and a
  `GROQ_API_KEY` (free tier, [console.groq.com](https://console.groq.com)) switches
  intent parsing to `src/agent/llm_intent_parser.py`, which calls Groq's
  `llama-3.1-8b-instant` model. This is strictly additive: on any failure (missing
  key, network error, timeout, malformed response) it automatically falls back to the
  verified regex parser with no crash and no behavior change — confirmed by testing
  both an unset key and a deliberately invalid one. Verified end-to-end: with the LLM
  parser enabled, all 7 example queries produced identical routing decisions
  (`query_type`, `flagged_count`) to the regex parser.
- **Agentic coding assistance used:** Claude Code (Anthropic) — used throughout
  development for implementation, debugging, precision/recall validation against the
  real dataset, and UI development.
- **Libraries:** pandas, numpy, scikit-learn, Streamlit, Altair (bundled with
  Streamlit), requests, python-dotenv — see `requirements.txt` for the authoritative list.

---

## 👥 Team

**White Hats**

- K Pradeep Kumar — kumarkpradeep2005@gmail.com
- P Vahran — vahranp@gmail.com

---

## ⚠️ Notes

- No proprietary, confidential, or copyrighted data used — all data is public/synthetic, sourced and cited above.
- This is a hackathon prototype, not a production-grade system, per scope guidance.
