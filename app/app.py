"""
Streamlit interface for the Agentic Personal Finance Tracker.

    streamlit run app/app.py

Two screens, and only ever one of them on the page at a time.

1. Landing -- what this is, and one way in: upload a statement, or load the
   sample. Once a file is accepted a Start button appears.
2. Chat -- the title, the conversation, and a way back. Nothing else.

The separation is enforced by `st.session_state.started` and an st.stop().
Without the gate both screens render into the same page: Streamlit runs the
script top to bottom, so the landing hero and the uploader are already on screen
by the time the conversation is drawn underneath them.

Every answer carries a "show the working" panel naming the tool that ran, the
arguments it was called with and the table it read. A finance assistant asking
to be trusted on numbers should be able to show where each one came from -- and
in a demo it converts "the model said so" into something a sceptic can check.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Streamlit puts the script's own folder on sys.path, not the working
# directory, so `from src...` fails without this.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd                                                # noqa: E402
import streamlit as st                                             # noqa: E402

from src import analytics                                          # noqa: E402
from src.agent import answer_stream, plan, prepare_dataframe       # noqa: E402
from src.features import load_transactions                         # noqa: E402
from src.llm import is_available                                   # noqa: E402

st.set_page_config(page_title="Agentic Personal Finance Tracker",
                   page_icon="◈", layout="centered",
                   initial_sidebar_state="collapsed")

DEMO_PATH = PROJECT_ROOT / "data" / "processed" / "transactions.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "sample_bank_export.csv"

# Hides the two sample-data shortcuts on the landing page ("Explore with sample
# data" and "or download the CSV first") so a live demo offers one path only:
# upload the file being presented. The code behind them is untouched -- set this
# back to True to restore both buttons exactly as they were.
SHOW_SAMPLE_BUTTONS = False

SUGGESTIONS = [
    "Where am I overspending?",
    "How does this month compare to last month?",
    "Are there any suspicious transactions?",
    "How much am I paying for subscriptions?",
]


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

st.markdown("""
<style>
  /* No sidebar anywhere, and no control to summon one */
  section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"] {display: none !important;}

  #MainMenu, footer {visibility: hidden;}
  header[data-testid="stHeader"] {background: transparent; height: 0;}

  .block-container {max-width: 700px; padding-top: 3rem; padding-bottom: 6rem;}

  .eyebrow {
      font-size: 0.72rem; font-weight: 600; color: #4F46E5;
      letter-spacing: 0.13em; text-transform: uppercase;
      margin-bottom: 0.85rem;
  }
  h1.hero {
      font-size: 2.5rem; font-weight: 700; letter-spacing: -0.038em;
      color: #0F172A; line-height: 1.1; margin: 0 0 0.9rem 0; padding: 0;
  }
  .lede {
      font-size: 1.04rem; color: #64748B; line-height: 1.65;
      margin: 0 0 2.4rem 0; max-width: 32rem;
  }
  .step-label {
      font-size: 0.72rem; font-weight: 600; color: #94A3B8;
      letter-spacing: 0.11em; text-transform: uppercase;
      margin: 0 0 0.6rem 0;
  }
  [data-testid="stFileUploaderDropzone"] {
      border: 1.5px dashed #D5D9E2; border-radius: 12px; background: #FCFCFE;
      transition: border-color .15s ease, background .15s ease;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
      border-color: #4F46E5; background: #FAFAFF;
  }
  .rule {
      display: flex; align-items: center; gap: 1rem; color: #CBD5E1;
      font-size: 0.72rem; margin: 2rem 0 1.6rem 0;
      text-transform: uppercase; letter-spacing: 0.12em;
  }
  .rule:before, .rule:after {content: ""; flex: 1; height: 1px; background: #ECEEF3;}

  .ready {
      font-size: 0.88rem; color: #047857; background: #ECFDF5;
      border: 1px solid #A7F3D0; border-radius: 10px;
      padding: 0.7rem 0.95rem; margin: 1rem 0 0.9rem 0;
  }

  /* Primary call to action, scoped so chat chips are unaffected */
  .st-key-cta div[data-testid="stButton"] > button {
      border-radius: 10px; border: 1px solid #4F46E5;
      background: #4F46E5; color: #FFFFFF;
      font-size: 0.95rem; font-weight: 500; padding: 0.72rem 1rem;
      text-align: center;
  }
  .st-key-cta div[data-testid="stButton"] > button:hover {
      background: #4338CA; border-color: #4338CA; color: #FFFFFF;
  }
  .st-key-cta div[data-testid="stDownloadButton"] > button {
      border-radius: 10px; border: 1px solid transparent; background: transparent;
      color: #64748B; font-size: 0.85rem; font-weight: 400;
      padding: 0.4rem 1rem; text-align: center;
  }
  .st-key-cta div[data-testid="stDownloadButton"] > button:hover {
      color: #4F46E5; background: #F6F6FE; border-color: transparent;
  }

  /* Quiet control in the chat header */
  .st-key-topbar div[data-testid="stButton"] > button {
      border: 1px solid #E6E8EF; background: #FFFFFF; color: #64748B;
      font-size: 0.78rem; font-weight: 400; padding: 0.32rem 0.7rem;
      text-align: center; border-radius: 8px;
  }
  .st-key-topbar div[data-testid="stButton"] > button:hover {
      border-color: #4F46E5; color: #4F46E5;
  }

  /* One-line summary of the loaded file, under the title */
  .summary {
      display: flex; flex-wrap: wrap; gap: 0.35rem 1.4rem;
      padding: 0 0 0.9rem 0; margin-bottom: 1.4rem;
      border-bottom: 1px solid #ECEEF3;
  }
  .summary span {font-size: 0.8rem; color: #94A3B8; white-space: nowrap;}
  .summary b {
      color: #0F172A; font-weight: 600; font-variant-numeric: tabular-nums;
      margin-right: 0.28rem;
  }
  .summary .pos b {color: #059669;}
  .summary .neg b {color: #DC2626;}

  [data-testid="stChatMessage"] {background: transparent; padding: 0.15rem 0; border: none;}
  [data-testid="stChatMessage"] p {font-size: 1rem; line-height: 1.65; color: #1E293B;}

  /* The message box has to look like a box before it is clicked. Streamlit's
     default only draws a border once the field has focus, so on a white page
     the one control the whole screen exists for was invisible until you
     happened to click it. Visible at rest, stronger when active. */
  [data-testid="stChatInput"] {
      border: 2px solid #6167DE !important;
      border-radius: 12px !important;
      background: #FFFFFF !important;
      transition: border-color .15s ease;
  }
  [data-testid="stChatInput"]:hover {border-color: #4F46E5 !important;}
  [data-testid="stChatInput"]:focus-within {border-color: #3C34C9 !important;}

  /* Streamlit nests its own bordered wrapper inside; without this the two
     borders sit a pixel apart and read as a double line. */
  [data-testid="stChatInput"] > div,
  [data-testid="stChatInput"] [data-baseweb="textarea"],
  [data-testid="stChatInput"] [data-baseweb="base-input"] {
      border: none !important;
      background: transparent !important;
      box-shadow: none !important;
  }
  [data-testid="stChatInput"] textarea::placeholder {color: #94A3B8;}

  div[data-testid="stButton"] > button {
      border-radius: 10px; border: 1px solid #E6E8EF; background: #FFFFFF;
      color: #334155; font-size: 0.88rem; font-weight: 400;
      padding: 0.6rem 1rem; text-align: left;
  }
  div[data-testid="stButton"] > button:hover {
      border-color: #4F46E5; color: #4F46E5; background: #FFFFFF;
  }

  details[data-testid="stExpander"] {
      border: 1px solid #EFEFF3; border-radius: 10px; background: #FCFCFD;
  }
  details[data-testid="stExpander"] summary {font-size: 0.82rem; color: #64748B;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return load_transactions(str(DEMO_PATH))


@st.cache_data(show_spinner="Reading your statement...")
def load_upload(raw: bytes, recategorise: bool = False) -> pd.DataFrame:
    return prepare_dataframe(pd.read_csv(io.BytesIO(raw)),
                             recategorise=recategorise)


def has_categories(raw: bytes) -> bool:
    head = pd.read_csv(io.BytesIO(raw), nrows=1)
    return any(c.strip().lower() == "category" for c in head.columns)


for key, default in (("started", False), ("use_demo", False),
                     ("raw", None), ("history", [])):
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------------------------------
# Screen 1 -- landing
# --------------------------------------------------------------------------

if not st.session_state.started:
    st.markdown(
        '<div class="eyebrow">Agentic Personal Finance Tracker</div>'
        '<h1 class="hero">Your bank statement,<br>finally answering back.</h1>'
        '<div class="lede">Upload a statement to categorise your spending, '
        'flag unusual transactions, and ask questions in plain English.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="step-label">Upload your statement</div>',
                unsafe_allow_html=True)

    upload = st.file_uploader("Bank statement CSV", type="csv",
                              label_visibility="collapsed")

    # Keep the bytes, not the widget. Once the chat screen replaces this one the
    # uploader is no longer drawn, and an undrawn widget holds no value.
    if upload is not None:
        st.session_state.raw = upload.getvalue()

    if st.session_state.raw is not None:
        try:
            # A file that already carries categories would otherwise skip the
            # classifier entirely, so offer to ignore them and predict instead.
            if has_categories(st.session_state.raw):
                st.session_state.recat = st.checkbox(
                    "Ignore the categories in my file and predict them",
                    value=st.session_state.get("recat", False),
                    help="Your file already has a category column. Tick this "
                         "to have the trained classifier assign all of them "
                         "from the descriptions instead.")

            ready = load_upload(st.session_state.raw,
                                st.session_state.get("recat", False))
            spent = analytics.spending(ready)["amount"].sum()
            st.markdown(f'<div class="ready">{len(ready):,} transactions read, '
                        f'{analytics.available_months(ready)[0]} to '
                        f'{analytics.available_months(ready)[-1]}. '
                        f'EGP {spent:,.2f} of spending.</div>',
                        unsafe_allow_html=True)
            with st.container(key="cta"):
                if st.button("Start", width="stretch"):
                    st.session_state.started = True
                    st.rerun()
        except Exception as exc:                                   # noqa: BLE001
            st.error(f"Could not read that file.\n\n{exc}")
            st.session_state.raw = None
    elif SHOW_SAMPLE_BUTTONS:
        st.markdown('<div class="rule">no statement handy</div>',
                    unsafe_allow_html=True)
        with st.container(key="cta"):
            left, mid, right = st.columns([1, 2.2, 1])
            with mid:
                if st.button("Explore with sample data", width="stretch"):
                    st.session_state.use_demo = True
                    st.session_state.started = True
                    st.rerun()
                if SAMPLE_PATH.exists():
                    st.download_button("or download the CSV first",
                                       data=SAMPLE_PATH.read_bytes(),
                                       file_name="sample_bank_export.csv",
                                       mime="text/csv", width="stretch")

    # Sits outside the branch above so it still shows when the sample-data
    # shortcuts are hidden -- it warns about routing, not about the sample.
    if st.session_state.raw is None and not is_available():
        st.caption("No API key found — questions will be routed by keyword "
                   "matching. Every calculation still works.")

    st.stop()


# --------------------------------------------------------------------------
# Screen 2 -- the conversation, and nothing else
# --------------------------------------------------------------------------

df = (load_demo() if st.session_state.use_demo
      else load_upload(st.session_state.raw,
                       st.session_state.get("recat", False)))

health = analytics.budget_health(df)
months = analytics.available_months(df)
net = health["net"]

with st.container(key="topbar"):
    title_col, reset_col = st.columns([4, 1])
    with title_col:
        st.markdown('<div class="eyebrow" style="margin-bottom:0.3rem;'
                    'padding-top:0.35rem;">Agentic Personal Finance Tracker'
                    '</div>', unsafe_allow_html=True)
    with reset_col:
        if st.button("Start over", width="stretch"):
            st.session_state.update(started=False, use_demo=False, raw=None,
                                    history=[], recat=False)
            st.rerun()

# A one-line summary of what was loaded, so the user can sanity-check that the
# right file went in before trusting a single answer about it.
st.markdown(
    f'<div class="summary">'
    f'<span><b>{health["transactions"]:,}</b> transactions</span>'
    f'<span><b>{len(months)}</b> months</span>'
    f'<span>{months[0]} to {months[-1]}</span>'
    f'<span><b>EGP {health["total_spending"]:,.0f}</b> spent</span>'
    f'<span><b>EGP {health["total_income"]:,.0f}</b> in</span>'
    f'<span class="{"pos" if net >= 0 else "neg"}">'
    f'<b>EGP {net:,.0f}</b> net</span>'
    f'</div>', unsafe_allow_html=True)

# The opening chips live in a slot we can wipe, so asking a question clears them
# in the same run rather than leaving them above the conversation.
opening = st.empty()
clicked = None
if not st.session_state.history:
    with opening.container():
        row1, row2 = st.columns(2), st.columns(2)
        for col, text in zip(list(row1) + list(row2), SUGGESTIONS):
            if col.button(text, width="stretch"):
                clicked = text

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander(f"Show the working — {entry['tool']}"):
            st.caption(f"Router: {entry['router']}  ·  "
                       f"Answer written by: {entry['answer_source']}  ·  "
                       f"Arguments: {entry['args'] or 'defaults'}")
            table = entry.get("table")
            if isinstance(table, pd.DataFrame) and not table.empty:
                st.dataframe(table, width="stretch")
            st.text(entry["data_text"])

typed = st.chat_input("Ask about your spending, or anything unusual...")
question = clicked or typed

if question:
    opening.empty()

    with st.chat_message("user"):
        st.write(question)

    entry = {"question": question, "answer": "", "tool": "—", "args": {},
             "router": "—", "answer_source": "error",
             "table": pd.DataFrame(), "data_text": ""}

    with st.chat_message("assistant"):
        try:
            with st.spinner(""):
                choice, result = plan(df, question)
            entry.update(tool=choice["tool"], args=choice["args"],
                         router=choice.get("router", "rules"),
                         table=result["table"], data_text=result["data_text"])

            # Stream the prose. Same total time, but the first words land in a
            # few hundred milliseconds instead of the reader watching a blank
            # box for a second and a half.
            try:
                text = st.write_stream(answer_stream(question, result))
                entry.update(answer=text, answer_source="llm")
            except Exception:                                      # noqa: BLE001
                text = f"{result['headline']}\n\n{result['data_text']}"
                st.write(text)
                entry.update(answer=text, answer_source="template")

        except Exception as exc:                                   # noqa: BLE001
            entry["answer"] = f"Something went wrong: {exc}"
            st.write(entry["answer"])

        with st.expander(f"Show the working — {entry['tool']}"):
            st.caption(f"Router: {entry['router']}  ·  "
                       f"Answer written by: {entry['answer_source']}  ·  "
                       f"Arguments: {entry['args'] or 'defaults'}")
            if isinstance(entry["table"], pd.DataFrame) and not entry["table"].empty:
                st.dataframe(entry["table"], width="stretch")
            st.text(entry["data_text"])

    st.session_state.history.append(entry)
