"""
CareVal AI — Streamlit UI.


Mobile-first, warm, sisterly. Centered ~500px. Valuation runs when
the user says the day is done — no extra confirmation screen.
Run: streamlit run src/app.py
"""
from __future__ import annotations

import sys
import uuid
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.agents.conversational_agent import ensure_gemini, generate, gemini_ready
from src.chat_engine import landing_close
from src.config import config
from src.graph.orchestrator import (
    is_done_request,
    is_duration_only,
    is_idea_request,
    is_question_about_results,
    is_reset_request,
    is_unsure_duration,
    parse_stated_minutes,
    run_turn,
)
from src.graph.state import LoggedTask
from src.utils.bigquery_client import persist_day_log
from src.utils.classifier import classify_task, keyword_classify
from src.utils.data_loader import load_taxonomy
from src.utils.logging_setup import setup_logging
from src.utils.sample_provider import build_sample_day_tasks, list_sample_days
from src.utils.valuation import (
    compute_mospi_comparisons,
    compute_valuation,
    format_duration,
    get_market_enquiries,
    get_ranked_categories,
)

setup_logging()

st.set_page_config(
    page_title="CareVal",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
  --ink: #E8F5EE;
  --paper: #07110E;
  --muted: #8BA396;
  --line: rgba(16,185,129,0.18);
  --accent: #34D399;
  --accent-soft: rgba(16,185,129,0.12);
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
  color-scheme: dark !important;
  overflow-x: hidden !important;
}
[data-testid="stAppViewContainer"], .stApp, [data-testid="stHeader"],
[data-testid="stMain"], [data-testid="stAppScrollToBottomContainer"] {
  background: #07110E !important;
  background-image: radial-gradient(ellipse 80% 50% at 50% -8%, rgba(16,185,129,0.22) 0%, transparent 55%) !important;
}
header[data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer, header [data-testid="stHeaderActionElements"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
.stAppDeployButton, [data-testid="stAppDeployButton"],
div[data-testid="stHeader"] {
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
  display: none !important;
}
section[data-testid="stSidebar"] { display: none !important; }
.stApp { font-family: "Inter", system-ui, sans-serif !important; color: var(--ink) !important; }
.block-container {
  max-width: 440px !important;
  padding: 1rem 1.15rem 8.5rem 1.15rem !important;
  font-family: "Inter", system-ui, sans-serif !important;
}
@media (max-width: 375px) {
  .block-container { padding-left: 0.9rem !important; padding-right: 0.9rem !important; }
}
h1,h2,h3,p,label,span,div { word-wrap: break-word; overflow-wrap: anywhere; }

.care-heading {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.7rem;
  font-weight: 700;
  background: linear-gradient(135deg, #34D399, #6EE7B7);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
  letter-spacing: -0.03em;
  text-align: center;
}
.care-date {
  text-align: center;
  color: var(--muted);
  font-size: 0.68rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin: 0.28rem 0 0 0;
  font-weight: 600;
}
.care-kicker {
  text-align: center;
  color: var(--muted);
  font-size: 0.92rem;
  margin: 0.25rem 0 0 0;
  font-weight: 400;
}

.care-progress { margin: 1.1rem 0 1.25rem 0; }
.care-progress-label {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.45rem;
}
.care-track {
  height: 3px;
  background: var(--line);
  border-radius: 99px;
  overflow: hidden;
}
.care-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 99px;
}
.care-steps {
  display: flex;
  justify-content: space-between;
  margin-top: 0.45rem;
  font-size: 0.7rem;
  font-weight: 500;
  color: #5F7A6C;
}
.care-steps .on { color: var(--ink); font-weight: 600; }

.care-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1.05rem 1.1rem;
  margin: 0.7rem 0;
}
.care-card h3 {
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.05rem;
  margin: 0 0 0.4rem 0;
  color: var(--ink);
  font-weight: 600;
}
.muted { color: var(--muted); font-size: 0.84rem; }

.task-line {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.45rem 0;
  border-bottom: 1px solid var(--line);
  font-size: 0.92rem;
  color: var(--ink);
}
.task-line:last-child { border-bottom: none; }
.disclaimer {
  font-size: 0.72rem;
  color: var(--muted);
  margin-top: 0.7rem;
  line-height: 1.45;
}

[data-testid="stChatMessage"] {
  background: transparent !important;
  padding: 0.15rem 0 0.55rem 0 !important;
  gap: 0 !important;
  border: none !important;
}
[data-testid="stChatMessage"] > div:first-child { display: none !important; }
[data-testid="stChatMessageContent"] {
  max-width: 100%;
  overflow-wrap: anywhere;
  font-size: 0.98rem;
  line-height: 1.55;
  color: var(--ink);
  padding: 0 !important;
}
[data-testid="stChatMessage"]:has([aria-label*="user"]) {
  flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] {
  background: #FFFFFF !important;
  border: 1px solid rgba(16,185,129,0.25);
  border-radius: 14px 14px 4px 14px;
  padding: 0.65rem 0.9rem;
  color: #07110E !important;
}
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] span,
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] li {
  color: #07110E !important;
}

div.stButton > button {
  border-radius: 12px !important;
  border: 1px solid var(--line) !important;
  background: rgba(255,255,255,0.04) !important;
  color: var(--ink) !important;
  font-weight: 550 !important;
  font-size: 0.86rem !important;
  padding: 0.62rem 0.45rem !important;
  box-shadow: none !important;
}
div.stButton > button:hover {
  border-color: rgba(16,185,129,0.45) !important;
  background: var(--accent-soft) !important;
  color: #6EE7B7 !important;
}
div.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  background: #10B981 !important;
  color: #042F22 !important;
  border-color: #10B981 !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  padding: 0.8rem 0.5rem !important;
}
div.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
  background: #34D399 !important;
}

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div {
  background: var(--paper) !important;
  border: none !important;
  box-shadow: none !important;
}
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] > div {
  border: 1px solid rgba(16,185,129,0.28) !important;
  background: #FFFFFF !important;
  border-radius: 14px !important;
  box-shadow: none !important;
  color-scheme: light !important;
}
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] [contenteditable="true"],
[data-testid="stChatInputTextArea"] {
  font-size: 0.96rem !important;
  color: #000 !important;
  -webkit-text-fill-color: #000 !important;
  background: #FFFFFF !important;
  caret-color: #000 !important;
}
div[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputTextArea"]::placeholder {
  color: #6B7280 !important;
  -webkit-text-fill-color: #6B7280 !important;
}
[data-testid="stChatInputSubmitButton"] {
  background: #10B981 !important;
  color: #042F22 !important;
  border-radius: 10px !important;
}

[data-testid="stExpander"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  margin-top: 0.65rem !important;
}
[data-testid="stExpander"] summary {
  color: var(--muted) !important;
  font-size: 0.82rem !important;
}

.thinking-state { color: var(--muted); font-size: 0.88rem; }
.thinking-state .thinking-label { color: var(--ink); margin-bottom: 0.3rem; }
.thinking-state .thinking-model { margin-top: 0.3rem; font-size: 0.74rem; color: var(--muted); }
.typing-dots span {
  display: inline-block;
  width: 0.35rem; height: 0.35rem; margin: 0 0.08rem;
  border-radius: 50%; background: var(--accent);
  animation: pulseDot 1.1s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.typing-dots span:nth-child(3) { animation-delay: 0.3s; }
@keyframes pulseDot {
  0%, 80%, 100% { opacity: 0.2; transform: translateY(0); }
  40% { opacity: 1; transform: translateY(-3px); }
}
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--muted) !important;
  text-align: center;
}
iframe, [data-testid="stIFrame"] { border: none !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border: none !important; }
</style>
"""

DISCLAIMER = (
    "Replacement cost estimate based on 2025–26 urban India market rates "
    "(platforms like Urban Company, UrbanPro, Justdial). "
    "Not an audited or government-certified figure."
)

CITY_CHIPS = ["Delhi", "Mumbai", "Bangalore", "Other"]
GENDER_CHIPS = [
    ("Woman", "female"),
    ("Man", "male"),
    ("Prefer not to say", "unspecified"),
]
GENDER_MAP = {label: value for label, value in GENDER_CHIPS}
AFFIRM = frozenset({"yes", "yeah", "yep", "sure", "ok", "okay", "please"})
AFFIRM_IDEA = AFFIRM | {"yes please"}
AFFIRM_MOSPI = AFFIRM | {"compare"}
DECLINE = frozenset({"no", "nope", "nah", "not now", "later"})

FOLLOW_UPS = {
    "311": "Was that breakfast, dinner, or both?",
    "312": "Was that a quick tidy, or a longer clean?",
    "313": "Was that washing, ironing, or putting clothes away?",
    "381": "Was that groceries, bills, or just keeping the house on track?",
    "411": "Which subject were you helping with?",
    "412": "Was that the school run, feeding, or just being with them?",
    "420": "Was that medicines, a doctor visit, or sitting with them?",
}

# What the user can type next — labels people actually say.
FLOW_STEPS = ("Start", "Log", "Value", "Next")

TASK_HINTS = [
    ("311", "cooking", "cooked for 45 min"),
    ("312", "cleaning", "cleaned for 30 min"),
    ("313", "laundry", "laundry for 20 min"),
    ("381", "errands", "groceries for 45 min"),
    ("411", "tutoring", "tutoring for 1 hour"),
    ("412", "childcare", "childcare for 2 hours"),
    ("420", "elder care", "looked after mum for 30 min"),
]


# ---- Session State ----

DEFAULT_SESSION_STATE = {
    "messages": [],
    "tasks": [],
    "ui_stage": "collecting",
    "user_gender": None,
    "user_persona": None,
    "user_city": None,
    "show_breakdown": False,
    "idea_index": 0,
    "research_consent": False,
    "bq_persisted": False,
    "welcomed": False,
    "pending_prompt": None,
    "awaiting_followup": False,
    "pending_user_text": None,
    "typing_ready": False,
    "sample_speaker": None,
    "awaiting_idea_offer": False,
    "show_idea": False,
    "awaiting_mospi_offer": False,
    "show_mospi": False,
    "greet_step": "name",
    "user_name": None,
}


def init_state() -> None:
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def format_inr(amount: float) -> str:
    n = int(round(amount))
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) <= 3:
        return f"{sign}₹{s}"
    last3, rest = s[-3:], s[:-3]
    parts: list[str] = []
    while rest:
        parts.append(rest[-2:])
        rest = rest[:-2]
    return f"{sign}₹{','.join(reversed(parts))},{last3}"


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def get_day_tasks() -> list[LoggedTask]:
    return [LoggedTask(**t) for t in st.session_state.tasks]


def reset_day(full: bool = True) -> None:
    keys_to_clear = [
        "tasks", "ui_stage", "user_gender", "user_persona", "user_city",
        "show_breakdown", "idea_index", "bq_persisted", "pending_prompt",
        "awaiting_followup", "pending_user_text", "typing_ready",
        "sample_speaker", "awaiting_idea_offer", "show_idea",
        "awaiting_mospi_offer", "show_mospi", "user_name"
    ]
    for key in keys_to_clear:
        st.session_state[key] = DEFAULT_SESSION_STATE[key]
    st.session_state.greet_step = "name" if full else None
    if full:
        st.session_state.messages = []
        st.session_state.welcomed = False
        st.session_state.research_consent = False


def get_taxonomy() -> dict:
    try:
        return load_taxonomy(str(config.taxonomy_path))
    except Exception:
        return {}


def logged_categories() -> set[str]:
    return {t.category for t in get_day_tasks() if t.category != "unclassified"}


def remaining_task_hints() -> str:
    """Point people at tasks they have not logged yet."""
    covered = logged_categories()
    unused = [item for item in TASK_HINTS if item[0] not in covered]
    if not unused:
        return "If that's everything, say **that's all** and I'll add it up."
    names = ", ".join(f"**{label}**" for _, label, _ in unused)
    sample = unused[0][2]
    return (
        f"You can add {names} — e.g. *{sample}* — or say **that's all**."
    )


# ---- Ideas ----

def cycle_idea() -> None:
    st.session_state.idea_index = int(st.session_state.get("idea_index") or 0) + 1


def ideas_for_day(day: list[LoggedTask]) -> list[dict]:
    """Ranked today categories first, then other paths — so 'Another idea' stays rich."""
    seen: set[str] = set()
    ordered: list[dict] = []
    taxonomy = get_taxonomy()
    ranked = get_ranked_categories(day)
    today_set = set(ranked)
    rest = [c for c in taxonomy if c not in today_set and c != "unclassified"]
    for code in ranked + rest:
        idea = taxonomy.get(code, {}).get("micro_business")
        if not isinstance(idea, dict) or not idea.get("title"):
            continue
        title = idea["title"]
        if title in seen:
            continue
        seen.add(title)
        ordered.append(
            {
                **idea,
                "from_today": code in today_set,
                "source_category": code,
                "source_label": taxonomy.get(code, {}).get("display_label") or code,
            }
        )
    return ordered


def current_idea() -> dict | None:
    ideas = ideas_for_day(get_day_tasks())
    if not ideas:
        ideas = ideas_for_day([])
    if not ideas:
        return None
    idx = int(st.session_state.get("idea_index") or 0) % len(ideas)
    return ideas[idx]


def enquiry_for_category(code: str | None) -> dict | None:
    if not code:
        return None
    entry = get_taxonomy().get(code) or {}
    enquiry = entry.get("market_enquiry")
    if not isinstance(enquiry, dict):
        return None
    links = [
        link
        for link in (enquiry.get("links") or [])
        if isinstance(link, dict) and link.get("label") and link.get("url")
    ]
    if not links:
        return None
    return {
        "icon": entry.get("icon", "✨"),
        "service": enquiry.get("service") or entry.get("display_label") or "Local service",
        "prompt": enquiry.get("prompt") or "Ask nearby what they charge.",
        "links": links,
    }


def format_enquiry_links(items: list[dict], *, limit: int = 2) -> str:
    blocks: list[str] = []
    for item in items[:limit]:
        link_md = " · ".join(
            f"[{link['label']}]({link['url']})" for link in item.get("links", [])[:2]
        )
        if not link_md:
            continue
        icon = item.get("icon") or "✨"
        service = item.get("service") or "this work"
        blocks.append(f"{icon} **{service}** — {link_md}")
    if not blocks:
        return ""
    return (
        "People nearby already pay for this. Have a look:\n\n"
        + "\n".join(blocks)
        + "\n\n_Listings only — not used in the rupee figure._"
    )


def idea_chat_line(idea: dict) -> str:
    title = idea.get("title") or "a small nearby experiment"
    step = idea.get("step_one") or ""
    if idea.get("from_today"):
        lead = "Based on what you've logged today, here's one path others like you have tried"
    else:
        lead = "Here's one path people nearby have tried"
    line = f"{lead}: **{title}**. {step}"
    enquiry = enquiry_for_category(idea.get("source_category"))
    extra = format_enquiry_links([enquiry] if enquiry else [])
    if extra:
        return f"{line}\n\n{extra}"
    return line


def maybe_warm_reply(user_text: str, fallback: str) -> str:
    try:
        ensure_gemini()
        if not gemini_ready():
            return fallback
        city = st.session_state.get("user_city")
        persona = st.session_state.get("user_persona")
        context = (
            f"City: {city or 'unknown'}. Persona: {persona or 'unknown'}. "
            f"Tasks logged: {len(st.session_state.get('tasks') or [])}."
        )
        reply = generate(
            "You are CareVal, a warm, concise sisterly voice for unpaid care work in India. "
            "Reply in 1–3 short sentences. Never invent a rupee figure. "
            "Do not sound like a chatbot. No bullet lists.",
            f"{context}\nUser: {user_text}",
        )
        return (reply or "").strip() or fallback
    except Exception:
        return fallback


def offer_idea(advance: bool = False) -> None:
    if advance:
        cycle_idea()
    idea = current_idea()
    if idea:
        add_message("assistant", idea_chat_line(idea))
    else:
        add_message(
            "assistant",
            "Log a little of today's work first — cooking, kids, housework — "
            "and I'll suggest a small path that uses those skills.",
        )


def gender_from_text(text: str) -> str | None:
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    if "woman" in lowered or "female" in lowered:
        return "Woman"
    if lowered in {"man", "male"} or lowered.startswith("man ") or "i'm a man" in lowered:
        return "Man"
    if any(w in lowered for w in ("skip", "prefer not", "rather not", "prefer not to say")):
        return "Prefer not to say"
    return None


def apply_gender_choice(label: str) -> None:
    value = GENDER_MAP.get(label)
    if value and value != "unspecified":
        st.session_state.user_gender = value
    if st.session_state.ui_stage == "results":
        if value == "unspecified" or not value:
            add_message("assistant", "That's completely fine. Ask me anything about the total.")
        else:
            add_message("assistant", "Thank you — I'll use that only for the MoSPI comparison.")
        return
    if value == "unspecified" or not value:
        add_message("assistant", "That's completely fine. What happened next in your day?")
    else:
        add_message("assistant", "Thank you — that helps me compare your time more fairly. What happened next?")


def mospi_spoken_line() -> str:
    rows = [r for r in compute_mospi_comparisons(get_day_tasks()) if r.user_minutes > 0]
    if not rows:
        return "I don't have enough of today's minutes yet to compare with the national average."
    top = max(rows, key=lambda r: r.user_minutes)
    return (
        f"{top.comparison_text} "
        "That's official time-use minutes — it never sets the rupee figure."
    )


# ---- Logging ----

def offer_mospi() -> None:
    st.session_state.show_mospi = True
    st.session_state.awaiting_mospi_offer = False
    add_message("assistant", mospi_spoken_line())


def format_day_tally() -> str:
    """Each chore stays its own line item — total is the sum, not a merge."""
    taxonomy = get_taxonomy()
    day = get_day_tasks()
    bits = []
    for task in day:
        label = taxonomy.get(task.category, {}).get("display_label") or task.normalized_task
        bits.append(f"{label} {format_duration(task.estimated_minutes)}")
    total = format_duration(sum(t.estimated_minutes for t in day))
    if len(bits) <= 1:
        return total
    return " + ".join(bits) + f" = {total}"


def commit_task_direct(phrase: str, minutes: int) -> None:
    """Log a task directly — no LLM pipeline needed when phrase+minutes are known."""
    ensure_gemini()
    category = classify_task(phrase, gemini_client=gemini_ready())
    task = LoggedTask(
        task_id=str(uuid.uuid4()),
        raw_text=phrase,
        normalized_task=phrase.lower()[:80],
        estimated_minutes=minutes,
        category=category,
    )
    st.session_state.tasks = st.session_state.tasks + [task.model_dump()]
    st.session_state.pending_prompt = None
    day = get_day_tasks()
    taxonomy = get_taxonomy()
    display = taxonomy.get(category, {}).get("display_label") or phrase.title()
    icon = taxonomy.get(category, {}).get("icon", "✨")
    if len(day) >= 2:
        follow = remaining_task_hints()
        st.session_state.awaiting_followup = False
    else:
        detail = FOLLOW_UPS.get(category)
        follow = f"{detail} {remaining_task_hints()}" if detail else remaining_task_hints()
        st.session_state.awaiting_followup = category in FOLLOW_UPS
    add_message(
        "assistant",
        f"I've got {icon} **{display}** for {format_duration(minutes)} — kept on its own. "
        f"Today: {format_day_tally()}. {follow}",
    )
    st.session_state.ui_stage = "collecting"


def commit_task_text(text: str) -> None:
    result = run_turn(
        {
            "latest_message": text,
            "day_tasks": st.session_state.tasks,
            "chat_history": st.session_state.messages,
        }
    )
    st.session_state.tasks = result.get("day_tasks", st.session_state.tasks)
    st.session_state.pending_prompt = None
    extracted = result.get("classified_tasks") or result.get("extracted_tasks") or []
    error = result.get("last_error")
    if error and not extracted:
        add_message("assistant", error)
        return
    if not extracted:
        add_message(
            "assistant",
            "I didn't quite catch that. Try a task plus a time — "
            f"{remaining_task_hints()}",
        )
        return
    bits = [
        f"**{item.get('normalized_task') or item.get('raw_text') or 'a task'}** "
        f"(~{item.get('estimated_minutes', 0)} min)"
        for item in extracted
    ]
    last_code = extracted[-1].get("category") if extracted else None
    if last_code in FOLLOW_UPS and len(get_day_tasks()) < 2:
        follow = f"{FOLLOW_UPS[last_code]} {remaining_task_hints()}"
        st.session_state.awaiting_followup = True
    else:
        follow = remaining_task_hints()
        st.session_state.awaiting_followup = False
    total = sum(t.estimated_minutes for t in get_day_tasks())
    add_message(
        "assistant",
        "That counts — "
        + "; ".join(bits)
        + f". So far that's {format_duration(total)}. {follow}",
    )
    st.session_state.ui_stage = "collecting"


def load_sample_day(sample_id: str) -> None:
    """Instantly load a precomputed realistic day without requiring typing or API calls."""
    reset_day(full=True)
    tasks, gender = build_sample_day_tasks(sample_id)
    if not tasks:
        return
    st.session_state.tasks = [t.model_dump() for t in tasks]
    if gender:
        st.session_state.user_gender = gender
    st.session_state.welcomed = True
    st.session_state.greet_step = None

    samples = list_sample_days()
    sample_info = next((s for s in samples if s.get("id") == sample_id), None) or {}
    speaker = sample_info.get("speaker") or "someone"
    pronoun = (sample_info.get("pronoun") or "they").lower()
    intro = sample_info.get("intro") or f"Here's what {speaker} told me about their day."
    st.session_state.sample_speaker = speaker
    verb = "She said" if pronoun == "she" else "He said" if pronoun == "he" else "They said"
    bullets = "\n".join(
        f"- {t.raw_text} (*{format_duration(t.estimated_minutes)}*)" for t in tasks
    )
    add_message("assistant", intro)
    add_message("assistant", f"{verb}:\n\n{bullets}")
    finish_day()


def finish_day() -> None:
    """End collecting and show results, with a chat reply."""
    if not get_day_tasks():
        add_message(
            "assistant",
            "We haven't logged anything yet — tell me one thing you did today first.",
        )
        return
    st.session_state.ui_stage = "results"
    st.session_state.show_idea = False
    st.session_state.awaiting_idea_offer = True
    line = landing_close(
        get_day_tasks(),
        speaker=st.session_state.get("sample_speaker") or st.session_state.get("user_name"),
    ).split("\n\n")[0]
    add_message(
        "assistant",
        f"{line}\n\nWould you like a suggestion for what this skill could become?",
    )


def maybe_capture_profile(text: str) -> None:
    """Quietly learn persona/city from free text when mentioned."""
    lowered = (text or "").lower()
    if not st.session_state.get("user_persona"):
        if any(w in lowered for w in ("homemaker", "housewife", "stay at home", "stay-at-home")):
            st.session_state.user_persona = "homemaker"
        elif any(w in lowered for w in ("part-time", "full-time", "i work", "office", "my job")):
            st.session_state.user_persona = "employed"
    if not st.session_state.get("user_city"):
        for city in CITY_CHIPS:
            if city.lower() in lowered:
                st.session_state.user_city = city
                break


def ask_how_long(phrase: str) -> None:
    st.session_state.pending_prompt = {"phrase": phrase}
    st.session_state.awaiting_followup = False
    add_message(
        "assistant",
        "That counts. About how long did that take — even a guess is fine?",
    )


def is_followup_reply(text: str) -> bool:
    """True for short answers to our last detail question (e.g. 'both')."""
    if not st.session_state.get("awaiting_followup"):
        return False
    if is_done_request(text) or is_idea_request(text) or is_reset_request(text):
        return False
    if keyword_classify(text):
        return False
    if parse_stated_minutes(text):
        return False
    words = (text or "").strip().split()
    return 0 < len(words) <= 10


def _first_name(text: str) -> str:
    raw = (text or "").strip().lower()
    for prefix in ("my name is ", "i'm ", "i am ", "this is "):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    token = raw.split()[0] if raw.split() else ""
    return token[:24].title() if token.isalpha() else ""


# ---- Conversation ----

def handle_greeting(text: str) -> bool:
    """Name → how was the day → chores. Returns True if this turn is done."""
    step = st.session_state.get("greet_step")
    if not step:
        return False
    if keyword_classify(text) or parse_stated_minutes(text) or is_done_request(text):
        st.session_state.greet_step = None
        return False
    if step == "name":
        name = _first_name(text)
        st.session_state.user_name = name or None
        st.session_state.greet_step = "day"
        hello = f"Nice to meet you, {name}." if name else "Nice to meet you."
        add_message("assistant", f"{hello} How did your day go?")
        return True
    if step == "day":
        st.session_state.greet_step = None
        name = st.session_state.get("user_name")
        who = f"{name}, what" if name else "What"
        add_message(
            "assistant",
            f"{who} chores did you do today — cooking, cleaning, kids, laundry? "
            "A time helps, like *cooked for 45 min*.",
        )
        return True
    return False


def process_user_text(text: str) -> None:
    maybe_capture_profile(text)

    if handle_greeting(text):
        return

    if is_done_request(text):
        finish_day()
        return

    if is_idea_request(text):
        offer_idea()
        return

    gender_label = gender_from_text(text)
    if gender_label:
        also_task = bool(keyword_classify(text) or parse_stated_minutes(text))
        if also_task:
            value = GENDER_MAP.get(gender_label)
            if value and value != "unspecified":
                st.session_state.user_gender = value
        else:
            apply_gender_choice(gender_label)
            return

    if is_followup_reply(text):
        st.session_state.awaiting_followup = False
        add_message(
            "assistant",
            f"Got it — {text.strip()}. {remaining_task_hints()}",
        )
        return

    pending = st.session_state.get("pending_prompt") or {}
    pending_phrase = pending.get("phrase") if isinstance(pending, dict) else None
    minutes = parse_stated_minutes(text, loose=bool(pending_phrase))
    code = keyword_classify(text)

    if pending_phrase:
        if minutes is not None:
            commit_task_direct(pending_phrase, minutes)
            return
        if is_unsure_duration(text) and not code:
            commit_task_direct(pending_phrase, 30)
            msgs = st.session_state.messages
            if msgs and msgs[-1]["role"] == "assistant":
                msgs[-1]["content"] += "\n\nI used 30 min as a guess — say a number if you want to change it."
            return
        if not code:
            add_message(
                "assistant",
                "Even a rough time works — 20 minutes, an hour, whatever feels right.",
            )
            return
        st.session_state.pending_prompt = None

    # If the user typed multiple tasks (e.g. separated by commas, 'and', newlines),
    # route to the full extractor pipeline so each task is parsed individually.
    is_multi_clause = any(sep in text for sep in (",", ";", "\n", " and ", " then "))

    if code and not is_multi_clause:
        if minutes is not None:
            commit_task_direct(text.strip().rstrip(".!?"), minutes)
        else:
            ask_how_long(text.strip().rstrip(".!?"))
        return

    if minutes is not None and is_duration_only(text):
        day = get_day_tasks()
        if day:
            last = day[-1]
            updated = [
                {**t, "estimated_minutes": minutes} if t["task_id"] == last.task_id else t
                for t in st.session_state.tasks
            ]
            st.session_state.tasks = updated
            total = sum(t.estimated_minutes for t in get_day_tasks())
            add_message(
                "assistant",
                f"Okay — I'll use **{minutes} min** for that last one "
                f"({format_duration(total)} so far). {remaining_task_hints()}",
            )
            return

    commit_task_text(text)


# ---- Render ----

def flow_step() -> int:
    if st.session_state.ui_stage == "results":
        return 4 if st.session_state.get("show_idea") else 3
    if st.session_state.get("tasks"):
        return 2
    return 1


def render_progress() -> None:
    step = flow_step()
    label = FLOW_STEPS[step - 1]
    width = int((step / len(FLOW_STEPS)) * 100)
    chips = "".join(
        f'<span class="{"on" if i <= step else ""}">{name}</span>'
        for i, name in enumerate(FLOW_STEPS, start=1)
    )
    st.markdown(
        f'<div class="care-progress">'
        f'<div class="care-progress-label">Step {step} of {len(FLOW_STEPS)} · {label}</div>'
        f'<div class="care-track"><div class="care-fill" style="width:{width}%"></div></div>'
        f'<div class="care-steps">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def render_hero(rupees: int, duration: str, city_note: str) -> None:
    st.components.v1.html(
        f"""
<div style="text-align:center;padding:1.4rem 0 0.4rem 0;font-family:'Inter',system-ui,sans-serif;">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&display=swap');
    html,body {{ margin:0; background:transparent; }}
  </style>
  <div style="font-size:0.68rem;letter-spacing:0.16em;text-transform:uppercase;font-weight:600;color:#6EE7B7;margin-bottom:0.45rem;">Your number</div>
  <div id="care-rupees" style="font-family:'Fraunces',Georgia,serif;font-size:4.4rem;font-weight:700;background:linear-gradient(135deg,#34D399,#A7F3D0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:0.95;letter-spacing:-0.04em;">₹0</div>
  <div style="margin-top:0.7rem;color:#8BA396;font-size:0.95rem;">{duration} of care &amp; domestic work{city_note}</div>
</div>
<script>
(function() {{
  const target = {rupees};
  const el = document.getElementById('care-rupees');
  function fmt(n) {{
    const sign = n < 0 ? '-' : '';
    const s = String(Math.abs(n));
    if (s.length <= 3) return sign + '₹' + s;
    const last3 = s.slice(-3);
    let rest = s.slice(0, -3);
    const parts = [];
    while (rest.length) {{ parts.push(rest.slice(-2)); rest = rest.slice(0, -2); }}
    return sign + '₹' + parts.reverse().join(',') + ',' + last3;
  }}
  const start = performance.now();
  const dur = 800;
  function tick(now) {{
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = fmt(Math.round(target * eased));
    if (t < 1) requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}})();
</script>
""",
        height=196,
    )


def render_results() -> None:
    day = get_day_tasks()
    if not day:
        st.session_state.ui_stage = "collecting"
        return

    valuation = compute_valuation(day)
    duration = format_duration(valuation.total_minutes_logged)
    rupees = int(round(valuation.daily_value_inr))
    city_note = f" in {st.session_state.user_city}" if st.session_state.get("user_city") else ""
    render_hero(rupees, duration, city_note)
    st.markdown(
        f'<div class="disclaimer" style="text-align:center;margin:0 0 0.9rem 0;">{DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("awaiting_idea_offer") and not st.session_state.get("show_idea"):
        yes, no = st.columns(2)
        if yes.button("Yes, suggest an idea", type="primary", use_container_width=True):
            st.session_state.awaiting_idea_offer = False
            st.session_state.show_idea = True
            offer_idea()
            st.rerun()
        if no.button("No, I'm good", use_container_width=True):
            st.session_state.awaiting_idea_offer = False
            st.session_state.awaiting_mospi_offer = True
            add_message(
                "assistant",
                "That's fine. Want to see how your time compares with other women in India?",
            )
            st.rerun()
        return

    if st.session_state.get("awaiting_mospi_offer") and not st.session_state.get("show_mospi"):
        if st.button("Compare my time", type="primary", use_container_width=True):
            offer_mospi()
            st.rerun()
        return

    if not st.session_state.get("show_idea") and not st.session_state.get("show_mospi"):
        return

    ideas = ideas_for_day(day)
    idea_count = len(ideas)
    idea_pos = int(st.session_state.get("idea_index") or 0) % idea_count if idea_count else 0
    idea = ideas[idea_pos] if ideas else None
    annual = format_inr(valuation.daily_value_inr * 365)
    skills = " · ".join(valuation.skills) if valuation.skills else ""

    # 3. One next step — purpose: unpaid skill → one tiny paid experiment
    if idea:
        from_today = idea.get("from_today")
        heading = "This skill already has a market" if from_today else "Another skill path"
        why = (
            "Purpose: the same work you already did unpaid is something neighbors pay for. "
            "One small experiment — not a job offer or income promise."
        )
        source = idea.get("source_label") or ""
        source_line = f'<div class="muted" style="margin-bottom:0.35rem;">From: {source}</div>' if source else ""
        skill_line = f'<div class="muted" style="margin-bottom:0.35rem;">Skills today: {skills}</div>' if skills and from_today else ""
        st.markdown(
            f'<div class="care-card">'
            f'<h3>{heading}</h3>'
            f'<div class="muted" style="margin-bottom:0.55rem;">{why}</div>'
            f"{source_line}{skill_line}"
            f'<div style="font-weight:600;color:#ECFDF5;">{idea.get("title", "")}</div>'
            f'<div style="margin-top:0.4rem;color:#8BA396;font-size:0.93rem;">{idea.get("step_one", "")}</div>'
            f'<div class="disclaimer">Idea {idea_pos + 1} of {idea_count} — tap below to see another path.</div></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Another idea",
            key="cycle_idea_btn",
            use_container_width=True,
        ):
            offer_idea(advance=True)
            st.rerun()
        if not st.session_state.get("show_mospi") and st.button(
            "Compare my time", use_container_width=True, key="mospi_after_idea"
        ):
            offer_mospi()
            st.rerun()

    if st.session_state.show_breakdown:
        rows = "".join(
            f"<div class='task-line'><span>{tv.icon} {tv.display_label}"
            f"<br><span class='muted'>{tv.benchmark_role} · {tv.minutes} min</span></span>"
            f"<span style='color:#34D399;font-weight:600;'>{format_inr(tv.value_inr)}</span></div>"
            for tv in valuation.task_values
        )
        st.markdown(
            f'<div class="care-card"><h3>How this adds up</h3>'
            f"{rows}"
            f'<div class="muted" style="margin-top:0.5rem;">If this day repeated for a year (illustrative): <span style="color:#34D399;font-weight:600;">{annual}</span></div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Ask nearby what this work costs", expanded=False):
            enquiries = get_market_enquiries(day)
            if enquiries:
                for item in enquiries:
                    st.markdown(f"**{item['icon']} {item['service']}** — {item['prompt']}")
                    for link in item["links"]:
                        st.markdown(f"- [{link['label']}]({link['url']})")
            else:
                st.caption("No enquiry links for today's tasks.")
            st.caption("Neighbourhood listings only. Not used in the rupee figure above.")
        with st.expander("How we calculated this", expanded=False):
            st.markdown(
                "Value = (minutes ÷ 60) × a 2025–26 market hourly rate for a comparable paid role. "
                "Rates are sourced from urban India platforms (Urban Company, UrbanPro, Justdial) and set "
                "conservatively at the lower-middle of the verified range. "
                "MoSPI time-use data is used only to compare minutes — never to set wages. "
                f"{valuation.rate_source_note}"
            )

    with st.expander("Privacy", expanded=False):
        st.session_state.research_consent = st.checkbox(
            "I agree to contribute anonymous session data for care economy research.",
            value=st.session_state.research_consent,
            key="research_consent_box",
        )
        st.caption("Your data, your control — this is stored under your session and you can clear it anytime.")

    if config.is_gcp and st.session_state.research_consent and not st.session_state.bq_persisted:
        ok = persist_day_log(
            user_gender=st.session_state.user_gender,
            total_minutes=valuation.total_minutes_logged,
            daily_value_inr=valuation.daily_value_inr,
            tasks=[t.model_dump() for t in day],
        )
        st.session_state.bq_persisted = True
        if ok:
            st.toast("Anonymous day summary saved. Thank you.")


def handle_results_chat(text: str) -> None:
    gender_label = gender_from_text(text)
    if gender_label:
        apply_gender_choice(gender_label)
        return
    lowered = (text or "").strip().lower()
    if is_reset_request(text):
        reset_day(full=True)
        return
    if st.session_state.get("awaiting_idea_offer"):
        if lowered in AFFIRM_IDEA:
            st.session_state.awaiting_idea_offer = False
            st.session_state.show_idea = True
            offer_idea()
            return
        if lowered in DECLINE:
            st.session_state.awaiting_idea_offer = False
            st.session_state.awaiting_mospi_offer = True
            add_message(
                "assistant",
                "That's fine. Want to see how your time compares with other women in India?",
            )
            return
    if st.session_state.get("awaiting_mospi_offer"):
        if lowered in AFFIRM_MOSPI:
            offer_mospi()
            return
        if lowered in DECLINE:
            st.session_state.awaiting_mospi_offer = False
            add_message("assistant", "Okay. Ask anytime, or say **new day**.")
            return
    if is_idea_request(text) or "another idea" in lowered:
        st.session_state.awaiting_idea_offer = False
        st.session_state.show_idea = True
        offer_idea(advance=True)
        return
    if (
        is_question_about_results(text)
        or "compare" in lowered
        or "mospi" in lowered
        or "average" in lowered
    ):
        offer_mospi()
        return
    if "breakdown" in lowered or "line by line" in lowered:
        st.session_state.show_breakdown = True
        add_message("assistant", "I've opened the line-by-line below.")
        return
    fallback = (
        "I'm still here. Say **compare my time**, **suggest ideas**, or **new day**."
    )
    add_message("assistant", maybe_warm_reply(text, fallback))


# ---- Page ----
st.markdown(APP_CSS, unsafe_allow_html=True)

_today = date.today()
_date_line = f"{_today.strftime('%A')}, {_today.day} {_today.strftime('%B %Y')}"
st.markdown('<div class="care-heading">CareVal</div>', unsafe_allow_html=True)
st.markdown(f'<div class="care-date">{_date_line}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="care-kicker">The value of invisible work</div>',
    unsafe_allow_html=True,
)
render_progress()

if not st.session_state.welcomed and not st.session_state.messages:
    add_message("assistant", "Hi — I'm CareVal. What's your name?")
    st.session_state.welcomed = True
    st.session_state.greet_step = "name"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

if st.session_state.get("pending_user_text"):
    model_name = config.gemini_model
    with st.chat_message("assistant", avatar="🌿"):
        st.markdown(
            f'<div class="thinking-state">'
            f'<div class="thinking-label">Thinking</div>'
            f'<div class="typing-dots"><span></span><span></span><span></span></div>'
            f'<div class="thinking-model">{model_name}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

stage = st.session_state.ui_stage

if not st.session_state.get("pending_user_text"):
    if stage == "results":
        render_results()
    elif st.session_state.get("greet_step") in ("name", "day"):
        st.markdown(
            '<div class="care-progress-label" style="margin: 0.4rem 0 0.55rem 0;">Or try a sample day</div>',
            unsafe_allow_html=True,
        )
        sample_options = [
            ("sample_1", "Urban homemaker"),
            ("sample_2", "Non-urban homemaker"),
        ]
        cols = st.columns(len(sample_options))
        for col, (sid, slabel) in zip(cols, sample_options):
            if col.button(slabel, key=f"btn_{sid}", use_container_width=True):
                load_sample_day(sid)
                st.rerun()
    elif stage == "collecting" and not st.session_state.get("pending_prompt"):
        st.markdown(
            '<div class="care-progress-label" style="margin: 0.4rem 0 0.55rem 0;">Tap a chore</div>',
            unsafe_allow_html=True,
        )
        covered = logged_categories()
        choices = [item for item in TASK_HINTS if item[0] not in covered] or TASK_HINTS
        cols = st.columns(2)
        for i, (code, label, _example) in enumerate(choices):
            if cols[i % 2].button(label.title(), key=f"chore_{code}", use_container_width=True):
                add_message("user", label)
                ask_how_long(label)
                st.rerun()

_step = st.session_state.get("greet_step")
_placeholder = (
    "Your name…"
    if _step == "name"
    else "How did your day go?"
    if _step == "day"
    else "A chore… e.g. cooked for 45 min"
)
user_input = st.chat_input(_placeholder)
if user_input:
    add_message("user", user_input)
    st.session_state.pending_user_text = user_input
    st.session_state.typing_ready = False
    st.rerun()

if st.session_state.get("pending_user_text") and not st.session_state.get("typing_ready"):
    st.session_state.typing_ready = True
    st.rerun()

if st.session_state.get("pending_user_text") and st.session_state.get("typing_ready"):
    pending_text = st.session_state.pending_user_text
    st.session_state.pending_user_text = None
    st.session_state.typing_ready = False
    if stage == "results":
        handle_results_chat(pending_text)
    else:
        process_user_text(pending_text)
    st.rerun()
