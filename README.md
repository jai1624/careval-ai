# CareVal AI 🌿

**Invisible Work → Recognition → Economic Value → Comparison → Possibility.**

CareVal AI is a warm, conversational app that helps people (especially those doing unpaid domestic and caregiving work) talk about their day, see that work remembered and organized, understand its illustrative economic value using MoSPI Time Use Survey benchmarks, see how their day compares to national averages, and discover one grounded micro-business possibility based on their dominant skill.

> "I told CareVal what I did today → it remembered it → showed me the value of my invisible work → showed me what those skills could become."

**Built for Patchamomma 2026.** 📄 [Documentation](#https://tinyurl.com/ydus4y5w) · ✍️ [Blog post](https://tinyurl.com/2kb9hfmz) · 🎥 [https://shorturl.at/vNgyr]

---

## Why this architecture

Gemini is deliberately kept **narrow**: it only extracts structured tasks from the user's latest message (and acts as a fallback classifier for ambiguous text). Every other piece of intelligence — memory, deduplication, classification-by-keyword, valuation math, MoSPI comparisons, and confirmation gating — is **deterministic Python**. This keeps the numbers trustworthy, reproducible, testable, and fast, and keeps the LLM footprint small and auditable.

No RAG, no vector DB, no embeddings, no autonomous agent loops. LangGraph is used only as a thin, linear `extract → classify → merge` pipeline for clean composition — not for agentic behavior.

![CareVal AI architecture: Gemini handles extraction and fallback classification only; every stage that produces a number is deterministic Python](assets/architecture_diagram.png)

---

## Project structure

```
src/
├── app.py                        # Streamlit UI (backup)
├── chainlit_app.py               # Chainlit UI — run this locally
├── chat_engine.py                # Shared conversation (no UI)
├── config.py                     # Environment loading (local/gcp), never crashes on missing config
├── agents/
│   └── conversational_agent.py   # Gemini client + strict task-extraction prompt/parsing
├── graph/
│   ├── orchestrator.py           # Lightweight LangGraph: extract -> classify -> merge
│   └── state.py                  # Pydantic models (ExtractedTask, LoggedTask) + graph state schema
├── utils/
│   ├── data_loader.py            # Dynamic CSV/JSON parsing -- no hardcoded activity codes anywhere
│   ├── classifier.py             # Keyword match -> Gemini fallback -> unclassified
│   ├── memory.py                 # Deterministic dedup, merging, corrections, running totals
│   ├── valuation.py              # Deterministic valuation + MoSPI comparison engine
│   ├── sample_provider.py        # 3 offline demo days, zero Gemini/GCP calls
│   ├── bigquery_client.py        # Optional, consent-gated, ADC-based persistence (never a blocker)
│   └── logging_setup.py          # Structured Cloud Logging in GCP mode, stdout fallback otherwise
└── data/
    ├── tus_activity_codes.csv    # Activity codes + keyword lists (source of truth for valid codes)
    ├── wage_equivalents.csv      # activity_code -> synthetic reference hourly wage
    ├── mospi_benchmarks.json     # LOCKED MoSPI minutes/day benchmarks (domestic & caregiving, kept separate)
    ├── taxonomy.json             # Internal display labels + micro-business ideas per activity code
    └── sample_days.json          # 3 precomputed demo days for zero-credential demos

tests/                            # pytest suite (see "Testing" below)
requirements.txt
.env.example
Dockerfile
```

---

## Running locally

```bash
cd careval-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GEMINI_API_KEY=<your key>
# ENVIRONMENT=local is the default -- no GCP credentials needed.

# Primary UI (chat-native)
chainlit run src/chainlit_app.py

# Backup UI
streamlit run src/app.py
```

In Chainlit, use a starter like **Working Mom's Day** or type `cooked for 45 min`. Streamlit remains as a backup.

---

## Running on GCP (Cloud Run)

```bash
# Authenticate with Application Default Credentials (no service-account JSON needed)
gcloud auth application-default login

export ENVIRONMENT=gcp
export GCP_PROJECT_ID=your-project-id

docker build -t careval-ai .
docker run -p 8080:8080 \
  -e ENVIRONMENT=gcp \
  -e GCP_PROJECT_ID=your-project-id \
  -v ~/.config/gcloud:/root/.config/gcloud \
  careval-ai

# Or deploy directly:
gcloud run deploy careval-ai \
  --source . \
  --set-env-vars ENVIRONMENT=gcp,GCP_PROJECT_ID=your-project-id \
  --allow-unauthenticated
```

In `gcp` mode:
- Gemini calls can optionally route through Vertex AI (`USE_VERTEX_AI=true`).
- BigQuery persistence is available, but **only fires if the user checks the consent box** in the sidebar, and any failure (missing dataset, auth, network) is logged and swallowed gracefully — it never crashes the app.
- Logs are sent to Cloud Logging; if that setup fails for any reason, the app falls back to stdout logging automatically.

---

## The valuation model (how the numbers work)

1. **Classification**: each logged task is matched against `tus_activity_codes.csv` keyword lists (longest match wins). If nothing matches, Gemini is asked to pick strictly from the dynamically loaded set of valid codes (max 2 retries). If that also fails, the task becomes `unclassified` and is excluded from valuation but still shown to the user, explained plainly.
2. **Task value** = `(minutes / 60) * base_hourly_rate_inr` (from `wage_equivalents.csv`, a synthetic reference wage per benchmark role — never real salary data).
3. **Daily value** = sum of all classified task values. 
4. **Illustrative annualized value** = `daily_value * 365`, always labeled exactly that — never "salary," "income," or "official."
5. **MoSPI comparison**: domestic and caregiving minutes are tracked and compared **separately**, against locked benchmark constants (Domestic: Female 289 / Male 88 min/day; Caregiving: Female 137 / Male 75 min/day) — they are never combined into a single figure.
6. Every valuation screen shows the mandatory disclaimer verbatim:
   > "Estimated using MoSPI Time Use Survey benchmarks and synthetic reference wage data for this demo. Not an audited or government-certified figure."

Valuation only ever fires on **explicit user request** ("💰 Show my valuation") followed by a confirmation step ("Is that about right?" → ✅/➕/✏️) — never automatically based on turn count.

---

## Testing

```bash
pytest tests/ -v
```

The suite (46 tests) covers:
- CSV/JSON loading and dynamic keyword parsing (no hardcoded codes)
- Pydantic validation (negative/zero/unrealistic durations, malformed input)
- Keyword + fallback classification behavior
- Task merging, deduplication, and in-place duration corrections
- Valuation math and separate domestic/caregiving MoSPI comparisons
- Extraction JSON parsing robustness (malformed LLM output never crashes)
- Full sample-day pipeline execution with **zero credentials**

Every module is written to fail soft: missing API keys, corrupt data files, malformed LLM JSON, and offline GCP services all degrade gracefully rather than crashing the app.

---

## Privacy

> "Your data, your control — this is stored under your session and you can clear it anytime."

By default, everything lives in session state only. BigQuery persistence is opt-in, consent-gated, and only available in `gcp` mode.
