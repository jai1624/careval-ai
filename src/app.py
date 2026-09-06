"""
CareVal AI — Streamlit UI.


Mobile-first, warm, sisterly. Centered ~500px. Valuation runs when
the user says the day is done — no extra confirmation screen.
Run: streamlit run src/app.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.config import config
from src.agents.conversational_agent import ensure_gemini, generate, gemini_ready
from src.graph.orchestrator import (
    is_idea_request,
    is_question_about_results,
    is_done_request,
    is_reset_request,
    is_unsure_duration,
    is_duration_only,
    parse_stated_minutes,
    run_turn,
)
from src.utils.classifier import keyword_classify
from src.graph.state import LoggedTask
from src.utils.bigquery_client import persist_day_log
from src.utils.data_loader import load_taxonomy
from src.utils.sample_provider import list_sample_days, build_sample_day_tasks
from src.utils.logging_setup import setup_logging
from src.utils.valuation import (
    compute_mospi_comparisons,
    compute_valuation,
    format_duration,
    get_market_enquiries,
    get_ranked_categories,
)

setup_logging()

st.set_page_config(
    page_title="CareVal AI",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
  color-scheme: dark !important;
  overflow-x: hidden !important;
}
[data-testid="stAppViewContainer"], .stApp, [data-testid="stHeader"],
[data-testid="stMain"], [data-testid="stAppScrollToBottomContainer"] {
  background: #0A0F1E !important;
  background-image: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(16,185,129,0.18) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(99,102,241,0.12) 0%, transparent 50%),
    linear-gradient(180deg, #0A0F1E 0%, #0D1424 100%) !important;
}
[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer, header [data-testid="stHeaderActionElements"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
  visibility: hidden !important;
  height: 0 !important;
  display: none !important;
}
.block-container {
  max-width: 440px !important;
  padding: 1.5rem 1.2rem 9rem 1.2rem !important;
  font-family: "Inter", system-ui, sans-serif !important;
}
@media (max-width: 375px) {
  .block-container { padding-left: 0.85rem !important; padding-right: 0.85rem !important; }
}
h1,h2,h3,p,label,span,div { word-wrap: break-word; overflow-wrap: anywhere; }

/* ── Header ── */
.care-heading {
  font-family: "Playfair Display", Georgia, serif;
  font-size: 2.2rem;
  font-weight: 700;
  background: linear-gradient(135deg, #10B981 0%, #34D399 40%, #6EE7B7 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
  letter-spacing: -0.02em;
  text-align: center;
  filter: drop-shadow(0 0 24px rgba(16,185,129,0.4));
}
.care-date {
  text-align: center;
  color: rgba(167,193,167,0.6);
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin: 0.4rem 0 0 0;
  font-weight: 500;
}
.care-kicker {
  text-align: center;
  color: rgba(209,213,219,0.7);
  font-size: 0.9rem;
  margin: 0.4rem 0 1.6rem 0;
  letter-spacing: 0.01em;
  font-weight: 300;
}

/* ── Cards ── */
.care-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 20px;
  padding: 1.3rem 1.4rem;
  margin: 0.85rem 0;
  box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  animation: fadeSlideIn 0.4s cubic-bezier(0.22,1,0.36,1) both;
}
.muted { color: rgba(156,163,175,0.8); font-size: 0.85rem; }

/* ── Task list ── */
.task-line {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  font-size: 0.93rem;
  color: #D1D5DB;
}
.task-line:last-child { border-bottom: none; }
.disclaimer {
  font-size: 0.74rem;
  color: rgba(156,163,175,0.6);
  font-style: italic;
  margin-top: 0.75rem;
  line-height: 1.5;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  padding: 0.3rem 0 0.65rem 0 !important;
  gap: 0.6rem !important;
  border: none !important;
}
[data-testid="stChatMessage"] > div:first-child {
  background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(52,211,153,0.15)) !important;
  color: #34D399 !important;
  box-shadow: 0 0 0 1px rgba(16,185,129,0.2), 0 0 12px rgba(16,185,129,0.12) !important;
  width: 1.75rem !important;
  height: 1.75rem !important;
  min-width: 1.75rem !important;
  font-size: 0.85rem !important;
  border-radius: 50% !important;
}
[data-testid="stChatMessageContent"] {
  max-width: 100%;
  overflow-wrap: anywhere;
  font-size: 1.02rem;
  line-height: 1.65;
  color: #E5E7EB;
}
[data-testid="stChatMessage"]:has([aria-label*="user"]) {
  flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] {
  background: #FFFFFF !important;
  border: 1px solid rgba(16,185,129,0.25);
  border-radius: 18px 18px 4px 18px;
  padding: 0.7rem 1rem;
  color: #000000 !important;
}
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] span,
[data-testid="stChatMessage"]:has([aria-label*="user"]) [data-testid="stChatMessageContent"] li {
  color: #000000 !important;
}

/* ── Buttons ── */
div.stButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  background: rgba(255,255,255,0.05) !important;
  color: #D1D5DB !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  padding: 0.55rem 0.5rem !important;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05) !important;
  transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
  backdrop-filter: blur(8px);
}
div.stButton > button:hover {
  border-color: rgba(16,185,129,0.4) !important;
  background: rgba(16,185,129,0.1) !important;
  color: #34D399 !important;
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 8px 20px rgba(16,185,129,0.15), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}
div.stButton > button:active {
  transform: scale(0.97) !important;
}
div.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #059669, #10B981) !important;
  color: #ECFDF5 !important;
  border-color: transparent !important;
  font-size: 0.96rem !important;
  font-weight: 600 !important;
  padding: 0.78rem 0.5rem !important;
  box-shadow: 0 4px 16px rgba(16,185,129,0.35), inset 0 1px 0 rgba(255,255,255,0.15) !important;
}
div.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
  background: linear-gradient(135deg, #047857, #059669) !important;
  box-shadow: 0 8px 24px rgba(16,185,129,0.45) !important;
  transform: translateY(-2px) !important;
}

/* ── Input bar ── */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div {
  background: #0A0F1E !important;
  background-image: linear-gradient(180deg, rgba(10,15,30,0) 0%, #0A0F1E 30%) !important;
  border: none !important;
  box-shadow: none !important;
}
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] > div {
  border: 1px solid rgba(16,185,129,0.25) !important;
  background: #FFFFFF !important;
  border-radius: 16px !important;
  box-shadow: 0 0 0 1px rgba(16,185,129,0.08), 0 8px 32px rgba(0,0,0,0.3) !important;
  color-scheme: light !important;
  backdrop-filter: blur(16px) !important;
}
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] [contenteditable="true"],
[data-testid="stChatInputTextArea"] {
  border-radius: 16px !important;
  font-size: 0.98rem !important;
  color: #000000 !important;
  -webkit-text-fill-color: #000000 !important;
  background: #FFFFFF !important;
  caret-color: #000000 !important;
}
div[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputTextArea"]::placeholder {
  color: #6B7280 !important;
  -webkit-text-fill-color: #6B7280 !important;
  opacity: 1 !important;
}
[data-testid="stChatInputSubmitButton"] {
  background: linear-gradient(135deg, #059669, #10B981) !important;
  color: #ECFDF5 !important;
  border-radius: 10px !important;
  box-shadow: 0 2px 8px rgba(16,185,129,0.3) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
  box-shadow: none !important;
  margin-top: 0.85rem !important;
  backdrop-filter: blur(8px);
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
[data-testid="stExpander"] summary {
  color: rgba(156,163,175,0.8) !important;
  font-size: 0.84rem !important;
  justify-content: center !important;
}

/* ── Typing indicator ── */
.thinking-state {
  color: #9CA3AF;
  font-size: 0.92rem;
}
.thinking-state .thinking-label {
  font-style: italic;
  color: #D1D5DB;
  margin-bottom: 0.35rem;
}
.thinking-state .thinking-model {
  margin-top: 0.35rem;
  font-size: 0.78rem;
  color: #6B7280;
}
.typing-dots span {
  display: inline-block;
  width: 0.4rem;
  height: 0.4rem;
  margin: 0 0.1rem;
  border-radius: 50%;
  background: #10B981;
  animation: pulseDot 1.1s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.typing-dots span:nth-child(3) { animation-delay: 0.3s; }
@keyframes pulseDot {
  0%, 80%, 100% { opacity: 0.2; transform: translateY(0) scale(0.8); }
  40% { opacity: 1; transform: translateY(-4px) scale(1); }
}

/* ── Animations ── */
@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Misc ── */
.stCaption, [data-testid="stCaptionContainer"] {
  color: rgba(156,163,175,0.7) !important;
  text-align: center;
}
</style>
""",
    unsafe_allow_html=True,
)

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
TASK_HINTS = [
    ("311", "cooking", "cooked for 45 min"),
    ("312", "cleaning", "cleaned for 30 min"),
    ("313", "laundry", "laundry for 20 min"),
    ("381", "errands", "groceries for 45 min"),
    ("411", "tutoring", "tutoring for 1 hour"),
    ("412", "childcare", "childcare for 2 hours"),
    ("420", "elder care", "looked after mum for 30 min"),
]


def init_state() -> None:
    defaults = {
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
    }
    for key, value in defaults.items():
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
    st.session_state.tasks = []
    st.session_state.ui_stage = "collecting"
    st.session_state.user_gender = None
    st.session_state.user_persona = None
    st.session_state.user_city = None
    st.session_state.gender_prompted_on_results = False
    st.session_state.show_breakdown = False
    st.session_state.idea_index = 0
    st.session_state.bq_persisted = False
    st.session_state.pending_prompt = None
    st.session_state.awaiting_followup = False
    st.session_state.pending_user_text = None
    st.session_state.typing_ready = False
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
    mapping = {chip: value for chip, value in GENDER_CHIPS}
    value = mapping.get(label)
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


def commit_task_direct(phrase: str, minutes: int) -> None:
    """Log a task directly — no LLM pipeline needed when phrase+minutes are known."""
    import uuid
    from src.utils.classifier import classify_task
    from src.graph.state import LoggedTask
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
    total = sum(t.estimated_minutes for t in day)
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
        f"I've got {icon} **{display}** for about {minutes} min "
        f"({format_duration(total)} so far). {follow}",
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

    samples = list_sample_days()
    sample_info = next((s for s in samples if s.get("id") == sample_id), None)
    label = sample_info.get("label", "Sample day") if sample_info else "Sample day"

    add_message(
        "assistant",
        f"Loaded **{label}** with {len(tasks)} care tasks logged.",
    )
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
    reply = "Okay — here's what today's care work adds up to."
    extra = format_enquiry_links(get_market_enquiries(get_day_tasks()))
    if extra:
        reply = f"{reply}\n\n{extra}"
    add_message("assistant", reply)


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
        f"That counts. About how long did that take — even a guess is fine?",
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


def process_user_text(text: str) -> None:
    maybe_capture_profile(text)

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
            mapping = {chip: value for chip, value in GENDER_CHIPS}
            value = mapping.get(gender_label)
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


def render_results() -> None:
    day = get_day_tasks()
    if not day:
        st.session_state.ui_stage = "collecting"
        return

    valuation = compute_valuation(day)
    comparisons = compute_mospi_comparisons(day)
    ideas = ideas_for_day(day)
    idea_count = len(ideas)
    idea_pos = int(st.session_state.get("idea_index") or 0) % idea_count if idea_count else 0
    idea = ideas[idea_pos] if ideas else None

    duration = format_duration(valuation.total_minutes_logged)
    annual = format_inr(valuation.daily_value_inr * 365)
    skills = " · ".join(valuation.skills) if valuation.skills else ""
    rupees = int(round(valuation.daily_value_inr))

    city_note = f" in {st.session_state.user_city}" if st.session_state.get("user_city") else ""

    st.components.v1.html(
        f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(16,185,129,0.25);border-radius:20px;padding:1.3rem 1.4rem;
  font-family:'Inter',system-ui,sans-serif;box-shadow:0 4px 24px rgba(0,0,0,0.4),0 0 0 1px rgba(16,185,129,0.1),inset 0 1px 0 rgba(255,255,255,0.06);
  backdrop-filter:blur(12px);animation:fadeSlideIn 0.4s cubic-bezier(0.22,1,0.36,1) both;">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@700&display=swap');
    @keyframes fadeSlideIn {{ from {{ opacity:0; transform:translateY(14px); }} to {{ opacity:1; transform:translateY(0); }} }}
    html,body {{ margin:0; background:transparent; color-scheme:dark; }}
  </style>
  <div style="text-align:left;color:rgba(167,193,167,0.6);font-size:0.68rem;letter-spacing:0.18em;text-transform:uppercase;margin:0 0 0.75rem 0;font-weight:500;">Today · remembered</div>
  <div style="margin-bottom:0.85rem;font-family:'Playfair Display',Georgia,serif;font-size:1.1rem;line-height:1.6;color:#D1FAE5;">
    You did a lot today. This work is quiet — and it counts.</div>
  <div style="color:rgba(156,163,175,0.7);font-size:0.82rem;margin-bottom:0.2rem;">Replacement value</div>
  <div id="care-rupees" style="font-family:'Playfair Display',Georgia,serif;font-size:2.8rem;font-weight:700;background:linear-gradient(135deg,#10B981,#34D399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.05;letter-spacing:-0.02em;">₹0</div>
  <div style="margin-top:0.3rem;color:#A7F3D0;font-size:0.95rem;">{duration} of care &amp; domestic work{city_note}</div>
  <div style="font-size:0.72rem;color:rgba(156,163,175,0.5);font-style:italic;margin-top:0.85rem;line-height:1.5;">{DISCLAIMER}</div>
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
  const dur = 720;
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
        height=248,
    )

    # 2. One compact comparison
    domestic = next((c for c in comparisons if c.domain == "domestic"), None)
    caregiving = next((c for c in comparisons if c.domain == "caregiving"), None)
    lines = []
    if domestic and domestic.user_minutes > 0:
        lines.append(
            f"Domestic: you <b>{domestic.user_minutes} min</b> · women {domestic.female_avg_minutes} · men {domestic.male_avg_minutes}"
        )
    if caregiving and caregiving.user_minutes > 0:
        lines.append(
            f"Caregiving: you <b>{caregiving.user_minutes} min</b> · women {caregiving.female_avg_minutes} · men {caregiving.male_avg_minutes}"
        )
    if lines:
        st.markdown(
            f'<div class="care-card" style="background:rgba(99,102,241,0.07);border-color:rgba(99,102,241,0.2);">'
            f'<div style="font-weight:600;margin-bottom:0.4rem;color:#C4B5FD;">📊 Compared with MoSPI</div>'
            f'<div style="color:#D1D5DB;font-size:0.92rem;">{"<br>".join(lines)}</div>'
            f'<div class="disclaimer">Time-use minutes only. Domestic and caregiving stay separate.</div></div>',
            unsafe_allow_html=True,
        )

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
            f'<div class="care-card" style="background:rgba(245,158,11,0.06);border-color:rgba(245,158,11,0.2);">'
            f'<div style="font-weight:600;margin-bottom:0.35rem;color:#FCD34D;">🚀 {heading}</div>'
            f'<div class="muted" style="margin-bottom:0.55rem;font-size:0.82rem;">{why}</div>'
            f"{source_line}{skill_line}"
            f'<div style="font-weight:600;color:#FEF3C7;">{idea.get("title", "")}</div>'
            f'<div style="margin-top:0.4rem;color:#D1D5DB;font-size:0.93rem;">{idea.get("step_one", "")}</div>'
            f'<div class="disclaimer">Idea {idea_pos + 1} of {idea_count} — tap below to see another path.</div></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "🚀 Another idea",
            key="cycle_idea_btn",
            use_container_width=True,
        ):
            offer_idea(advance=True)
            st.rerun()

    if (
        not st.session_state.user_gender
        and not st.session_state.get("gender_prompted_on_results")
    ):
        st.session_state.gender_prompted_on_results = True
        add_message(
            "assistant",
            "Optional — if you're comfortable saying woman / man / prefer not to say, "
            "I can personalise the MoSPI comparison. Or just ask me anything about the total.",
        )

    if st.session_state.show_breakdown:
        rows = "".join(
            f"<div class='task-line'><span style='color:#E5E7EB;'>{tv.icon} {tv.display_label}"
            f"<br><span class='muted'>{tv.benchmark_role} · {tv.minutes} min</span></span>"
            f"<span style='color:#34D399;font-weight:600;'>{format_inr(tv.value_inr)}</span></div>"
            for tv in valuation.task_values
        )
        st.markdown(
            f'<div class="care-card"><div style="font-weight:600;margin-bottom:0.4rem;color:#E5E7EB;">How this adds up</div>'
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
    lowered = (text or "").lower()
    if is_idea_request(text) or "another idea" in lowered:
        offer_idea(advance=True)
        return
    if is_reset_request(text):
        reset_day(full=True)
        return
    if any(w in lowered for w in ("see details", "show details", "breakdown", "line by line")):
        st.session_state.show_breakdown = True
        add_message("assistant", "I've opened the line-by-line below.")
        return
    if is_question_about_results(text):
        st.session_state.show_breakdown = True
        fallback = (
            "The rupee figure is replacement cost: (minutes ÷ 60) × a conservative "
            "2025–26 market hourly rate for a comparable paid role. "
            "MoSPI compares minutes only — it never sets the wage. "
            "I've opened the line-by-line below."
        )
        add_message("assistant", maybe_warm_reply(text, fallback))
        return
    fallback = (
        "I'm still here. Ask how this was calculated, say **suggest ideas**, "
        "or **new day** when you want a fresh start."
    )
    add_message("assistant", maybe_warm_reply(text, fallback))


# ---- Header ----
_today = date.today()
_date_line = f"{_today.strftime('%A')}, {_today.day} {_today.strftime('%B %Y')}"
st.markdown('<div class="care-heading">CareVal AI</div>', unsafe_allow_html=True)
st.markdown(f'<div class="care-date">{_date_line}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="care-kicker">Recognising the value of invisible work</div>',
    unsafe_allow_html=True,
)

if not st.session_state.welcomed and not st.session_state.messages:
    add_message(
        "assistant",
        "Hi 🌿 I'm here to recognise the care work you do every day.\n\n"
        "Tell me one thing, with a time if you know it — "
        "*cooked for 45 min*, *childcare for 2 hours*, *tutoring for 1 hour*, "
        "*laundry for 20 min*.\n\n"
        "We'll go one task at a time. When you're ready, say **that's all**.",
    )
    st.session_state.welcomed = True

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
    elif not st.session_state.tasks and len(st.session_state.messages) <= 1:
        st.markdown(
            '<div style="margin: 0.8rem 0 0.4rem 0; font-size: 0.82rem; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em;">'
            '⚡ Try a sample day (instant demo)'
            '</div>',
            unsafe_allow_html=True,
        )
        sample_options = [
            ("sample_1", "👩 Working Mom's Day"),
            ("sample_2", "👵 Caring for Elders"),
            ("sample_3", "👨 Father Sharing Load"),
        ]
        cols = st.columns(len(sample_options))
        for col, (sid, slabel) in zip(cols, sample_options):
            if col.button(slabel, key=f"btn_{sid}", use_container_width=True):
                load_sample_day(sid)
                st.rerun()

user_input = st.chat_input("Type your day… e.g. cooked for 45 min")
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
