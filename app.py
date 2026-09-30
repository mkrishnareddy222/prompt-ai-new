"""
Groq Chatbot — Streamlit UI.

Run:

    streamlit run app.py
"""

import streamlit as st

from application.chat_service import ChatService
from app.config.settings import settings
from domain.models import ChatRequest, Message
from infrastructure.provider_factory import create_llm_provider


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Emma AI",
    page_icon="🤖",
    layout="wide",
)

theme_mode = st.session_state.get("theme_mode", "Dark")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --emma-ink: #202824;
        --emma-muted: #78817c;
        --emma-line: #e5e9e6;
        --emma-paper: #fbfcfa;
        --emma-panel: #f2f5f2;
        --emma-green: #28765d;
        --emma-green-soft: #e7f1eb;
        --emma-coral: #c8795e;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--emma-ink);
        letter-spacing: 0;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--emma-paper);
    }

    [data-testid="stApp"] {
        background: var(--emma-paper);
    }

    #MainMenu, footer {
        visibility: hidden;
        height: 0;
    }

    [data-testid="stHeader"] {
        visibility: visible !important;
        height: 3rem;
        background: transparent;
        pointer-events: none;
    }

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
        z-index: 1001;
        pointer-events: auto;
    }

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button {
        visibility: visible !important;
        color: var(--emma-ink) !important;
        background: var(--emma-panel) !important;
        border: 1px solid var(--emma-line) !important;
        border-radius: 9px;
    }

    [data-testid="stSidebar"] {
        background: #f1f4f1;
        border-right: 1px solid var(--emma-line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 0.9rem 0.95rem;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }

    [data-testid="stSidebar"] [data-testid="stSegmentedControl"] {
        width: 100%;
    }

    [data-testid="stSidebar"] hr {
        margin: 0.25rem 0 0.45rem;
    }

    [data-testid="stSidebar"] h2 {
        font: 700 0.78rem 'Manrope', sans-serif;
        letter-spacing: 0;
        text-transform: uppercase;
        color: var(--emma-muted);
    }

    [data-testid="stSidebar"] h3 {
        color: var(--emma-ink);
        font: 700 1rem 'Manrope', sans-serif;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #59645e;
        font-size: 0.88rem;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        border: 1px solid #dce3dd;
        border-radius: 10px;
        background: #fbfcfa;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] *,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        color: var(--emma-ink) !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: var(--emma-green);
    }

    .block-container {
        max-width: 980px;
        padding: 0.7rem 1.5rem 2.5rem;
    }

    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }

    .emma-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.05rem 0 0.45rem;
        margin-bottom: 0.45rem;
        border-bottom: 1px solid var(--emma-line);
        animation: arrive 420ms ease-out both;
    }

    .emma-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font: 800 1.08rem 'Manrope', sans-serif;
        color: var(--emma-ink);
    }

    .emma-mark {
        display: grid;
        width: 30px;
        height: 30px;
        place-items: center;
        border-radius: 11px;
        color: white;
        background: var(--emma-green);
        font-size: 1rem;
    }

    .emma-brand span {
        color: var(--emma-muted);
        font-weight: 500;
    }

    .emma-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--emma-muted);
        font-size: 0.8rem;
    }

    .emma-status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--emma-green);
        box-shadow: 0 0 0 3px var(--emma-green-soft);
    }

    .emma-welcome {
        max-width: 670px;
        margin: clamp(0.35rem, 1.2vh, 0.75rem) auto 0.35rem;
        text-align: center;
        animation: arrive 560ms ease-out both;
    }

    .emma-eyebrow {
        margin-bottom: 0.15rem;
        color: var(--emma-green);
        font: 700 0.72rem 'Manrope', sans-serif;
        text-transform: uppercase;
    }

    .emma-welcome h1 {
        margin: 0;
        color: var(--emma-ink);
        font: 700 clamp(1.65rem, 3.2vw, 2.2rem)/1.1 'Manrope', sans-serif;
        letter-spacing: 0;
    }

    .emma-welcome p {
        margin: 0.25rem 0 0;
        font-size: 0.9rem;
        color: var(--emma-muted);
        font-size: 1rem;
    }

    .emma-suggestions-label {
        margin: 0.35rem 0 0.2rem;
        color: var(--emma-muted);
        font-size: 0.78rem;
        text-align: left;
    }

    [data-testid="stButton"] button {
        min-height: 2rem;
        height: auto;
        border: 1px solid var(--emma-line);
        border-radius: 11px;
        color: #39443e;
        background: #fff;
        font-size: 0.85rem;
        white-space: normal;
        transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
    }

    [data-testid="stButton"] button p {
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
        line-height: 1.35;
    }

    [data-testid="stButton"] button:hover {
        border-color: #a8c3b5;
        background: #f5faf6;
        color: var(--emma-ink);
        transform: translateY(-1px);
    }

    [data-testid="stChatMessage"] {
        margin: 0.4rem 0;
        padding: 0.7rem 0.95rem;
        border: 1px solid var(--emma-line);
        border-radius: 15px;
        background: #fff;
        animation: arrive 220ms ease-out both;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.7;
        color: var(--emma-ink) !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
        color: var(--emma-ink) !important;
    }

    [data-testid="stChatInput"] {
        padding-top: 0.15rem;
    }

    [data-testid="stBottom"] {
        background: linear-gradient(180deg, rgba(251, 252, 250, 0), var(--emma-paper) 24%);
    }

    [data-testid="stBottom"] > div {
        background: transparent;
    }

    [data-testid="stChatInput"] textarea {
        min-height: 2.5rem;
        border: 1px solid #dce3dd;
        border-radius: 16px;
        background: #fff;
        color: var(--emma-ink) !important;
        caret-color: var(--emma-ink);
        box-shadow: 0 8px 26px rgba(31, 52, 42, 0.07);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--emma-muted) !important;
        opacity: 1;
    }

    [data-testid="stChatInput"] > div {
        border-radius: 16px;
        background: var(--emma-paper) !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #82aa96;
        box-shadow: 0 0 0 3px rgba(40, 118, 93, 0.11);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--emma-muted);
    }

    @keyframes arrive {
        from { opacity: 0; transform: translateY(7px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 0.55rem 0.8rem 4rem;
        }

        .emma-topbar {
            margin-bottom: 0.6rem;
            padding-bottom: 0.55rem;
        }

        .emma-welcome {
            margin: 1rem auto 0.6rem;
        }

        .emma-welcome h1 {
            font-size: 1.8rem;
        }

        [data-testid="stChatMessage"] {
            padding: 0.65rem 0.75rem;
        }
    }

    @media (max-height: 600px) {
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }

        .emma-topbar {
            margin-bottom: 0.25rem;
        }

        .emma-welcome p {
            display: none;
        }

        .emma-suggestions-label {
            margin-top: 0.15rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if theme_mode == "Dark":
    st.markdown(
        """
        <style>
        :root {
            --emma-ink: #e8efea;
            --emma-muted: #a2b0a7;
            --emma-line: #35433a;
            --emma-paper: #111814;
            --emma-panel: #1a241e;
            --emma-green: #83c6a2;
            --emma-green-soft: #20382b;
            --emma-coral: #e29a7d;
        }

        [data-testid="stApp"],
        [data-testid="stAppViewContainer"] {
            color: var(--emma-ink);
            background: var(--emma-paper) !important;
        }

        [data-testid="stSidebar"] {
            background: #17201a !important;
            border-color: var(--emma-line);
        }

        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button {
            color: var(--emma-ink) !important;
            background: #202b24 !important;
            border-color: var(--emma-line) !important;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #c5d0c8;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
        [data-testid="stSidebar"] [data-testid="stTextInput"] input {
            color: var(--emma-ink) !important;
            background: #202b24 !important;
            border-color: var(--emma-line) !important;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] * {
            color: var(--emma-ink) !important;
        }

        [data-testid="stChatMessage"] {
            color: var(--emma-ink);
            background: #1a241e !important;
            border-color: var(--emma-line);
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
            color: var(--emma-ink) !important;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--emma-ink) !important;
            background: #1a241e !important;
            border-color: var(--emma-line) !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--emma-muted) !important;
        }

        [data-testid="stChatInput"] > div,
        [data-testid="stBottom"] > div {
            background: var(--emma-paper) !important;
        }

        [data-testid="stBottom"] {
            background: linear-gradient(180deg, rgba(17, 24, 20, 0), var(--emma-paper) 24%) !important;
        }

        [data-testid="stButton"] button {
            color: var(--emma-ink) !important;
            background: #1a241e !important;
            border-color: var(--emma-line) !important;
        }

        [data-testid="stButton"] button:hover {
            background: #22332a !important;
            border-color: #527662 !important;
        }

        [data-testid="stSegmentedControl"] {
            background: #202b24;
        }

        [data-testid="stSegmentedControl"] button {
            color: var(--emma-ink);
        }

        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] {
            background: #202b24;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Dependency initialization
# ============================================================

def get_chat_service(
    provider: str,
    api_key_override: str | None,
) -> ChatService:
    llm_client = create_llm_provider(
        provider,
        api_key_override=api_key_override,
    )

    return ChatService(
        llm_client=llm_client
    )


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("### Your workspace")

    st.segmented_control(
        "Theme",
        options=["Light", "Dark"],
        default=theme_mode,
        key="theme_mode",
        help="Choose light or dark appearance.",
    )

    st.divider()
    st.header("Preferences")

    providers = ["groq", "openai", "gemini"]
    default_provider = (
        settings.LLM_PROVIDER
        if settings.LLM_PROVIDER in providers
        else "groq"
    )
    provider = st.selectbox(
        "AI provider",
        providers,
        index=providers.index(default_provider),
        format_func=lambda value: value.title(),
    )

    api_key_override = st.text_input(
        f"{provider.title()} API token",
        type="password",
        key=f"api_token_{provider}",
        help="Used for this session only. Leave blank to use the token from .env.",
    )

    with st.expander("Generation settings", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=settings.TEMPERATURE,
            step=0.1,
        )

        top_p = st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=settings.TOP_P,
            step=0.05,
            help="Controls nucleus sampling. Lower values focus on more likely tokens.",
        )

        max_tokens = st.slider(
            "Max tokens",
            min_value=128,
            max_value=4096,
            value=settings.MAX_TOKENS,
            step=128,
        )

        remember = st.toggle(
            "Remember conversation",
            value=False,
            help=(
                "Off = each question is sent alone. "
                "On = previous conversation is included."
            ),
        )

    st.caption(
        f"Model: `{settings.provider_model(provider)}`"
    )


try:
    chat_service = get_chat_service(provider, api_key_override)
except ValueError as exc:
    st.sidebar.error(str(exc))
    st.stop()


# ============================================================
# Header
# ============================================================

st.markdown(
    f"""
    <div class="emma-topbar">
      <div class="emma-brand"></div>    <span>AI</span></div>
      <div class="emma-status"><span class="emma-status-dot"></span>{provider.title()} · {settings.provider_model(provider)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Render conversation
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if not st.session_state.messages:
    st.markdown(
        """
                <div class="emma-welcome">
                    <div class="emma-eyebrow">Emma AI</div>
          <h1>What’s on your mind?</h1>
          <p>Bring a question, a rough idea, or something you want to work through.</p>
        </div>
        <div class="emma-suggestions-label">A place to start</div>
        """,
        unsafe_allow_html=True,
    )

    starter_prompts = [
        "Help me plan a focused week",
        "Explain a tricky idea simply",
        "Improve a draft I am working on",
    ]
    prompt_columns = st.columns(3)
    for column, starter_prompt in zip(prompt_columns, starter_prompts):
        with column:
            if st.button(starter_prompt, use_container_width=True):
                st.session_state.prompt_suggestion = starter_prompt
                st.rerun()


# ============================================================
# User input
# ============================================================

prompt = st.chat_input("  Message Emma...")
prompt = prompt or st.session_state.pop("prompt_suggestion", None)

if prompt:

    # --------------------------------------------
    # Display user message
    # --------------------------------------------

    with st.chat_message("user"):
        st.markdown(prompt)

    # --------------------------------------------
    # Convert Streamlit history into domain models
    # --------------------------------------------

    history = [
        Message(
            role=message["role"],
            content=message["content"],
        )
        for message in st.session_state.messages
    ]

    # --------------------------------------------
    # Create request
    # --------------------------------------------

    request = ChatRequest(
        message=prompt,
        remember=remember,
    )

    # --------------------------------------------
    # Save user message
    # --------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # --------------------------------------------
    # Generate response
    # --------------------------------------------

    with st.chat_message("assistant"):

        try:

            reply = st.write_stream(
                chat_service.stream(
                    request=request,
                    history=history,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply,
                }
            )

        except Exception as exc:

            st.error(
                f"⚠️ Unable to generate response: {exc}"
            )
