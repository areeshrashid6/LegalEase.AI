import streamlit as st

from openai import (
    OpenAI,
    AuthenticationError,
    APIConnectionError,
    APIError,
    RateLimitError,
)

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LegalEase AI",
    page_icon="⚖️",
    layout="wide",
)


# =========================================================
# LEGAL AI SYSTEM PROMPT
# =========================================================

LEGAL_PROMPT = """You are LegalEase AI, a domain-specific legal information assistant.

Your job is ONLY to provide general legal information.

You can help with:
- Legal terminology
- Contracts
- Legal documents
- Basic legal rights
- Employment law
- Consumer rights
- Property and rental law
- Business law
- Family law
- Intellectual property
- Legal procedures
- Courts and disputes
- Preparing questions for a lawyer
- Basic legal document classification

If a question is not substantially related to law or legal information,
do not answer it.

Instead reply:

"I don't have training for that topic. LegalEase AI is designed specifically
for legal-information questions. Please ask me about legal terminology,
contracts, rights, legal procedures, or another legal topic."

Provide general legal information only, not legal advice.

Laws can vary depending on country, state, province, or jurisdiction.
When jurisdiction matters, ask the user which jurisdiction applies.

Never invent:
- Laws
- Cases
- Legal citations
- Deadlines
- Penalties
- Court procedures
- Legal requirements

Explain legal concepts in simple everyday language.
Avoid unnecessary legal jargon.
"""


NONLEGAL = (
    "I don't have training for that topic. LegalEase AI is designed specifically "
    "for legal-information questions. Please ask me about legal terminology, "
    "contracts, rights, legal procedures, or another legal topic."
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "messages": [],
    "api_key": "",
    "model": "gpt-4o-mini",
    "connected": False,
    "connection_error": "",
    "focus": "General Legal Information",
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');


/* ---------------------------------------------------------
   GLOBAL
--------------------------------------------------------- */

html,
body,
[class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7f8fa;
    color: #182230;
}

.block-container {
    max-width: 1120px;
    padding: 55px 46px 70px;
}


/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #eef1f5;
    border-right: 1px solid #dfe4ea;
}

section[data-testid="stSidebar"] > div {
    padding: 28px 20px;
}


/* ---------------------------------------------------------
   BRAND
--------------------------------------------------------- */

.brand {
    padding: 4px 2px 24px;
}

.brand-mark {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: #182230;
    color: white;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 21px;
    margin-bottom: 12px;
}

.brand-name {
    font-family: 'Playfair Display', serif;
    font-size: 25px;
    font-weight: 700;
    color: #182230;
}

.brand-sub {
    color: #77808d;
    font-size: 11px;
    margin-top: 3px;
}


/* ---------------------------------------------------------
   SIDEBAR LABEL
--------------------------------------------------------- */

.sidebar-label {
    color: #8a94a2;
    font-size: 10px;
    font-weight: 700;

    letter-spacing: 1.3px;
    text-transform: uppercase;

    margin: 16px 0 8px;
}


/* ---------------------------------------------------------
   CONNECTION
--------------------------------------------------------- */

.connection {
    background: #f7fcf8;
    border: 1px solid #d5e8dc;
    border-radius: 11px;

    padding: 9px 11px;

    color: #52605a;
    font-size: 12px;

    margin-top: 10px;
}

.dot {
    display: inline-block;

    width: 7px;
    height: 7px;

    background: #42a36a;

    border-radius: 50%;

    margin-right: 7px;
}


/* ---------------------------------------------------------
   API ERROR
--------------------------------------------------------- */

.api-error {
    background: #fff6f5;

    border: 1px solid #f0c8c3;

    color: #9b332a;

    border-radius: 11px;

    padding: 11px;

    margin-top: 10px;

    font-size: 12px;

    line-height: 1.5;
}


/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.hero {
    text-align: center;

    padding: 20px 20px 30px;
}

.hero-kicker {
    color: #b08d57;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.8px;

    margin-bottom: 13px;
}

.hero h1 {
    font-family: 'Playfair Display', serif;

    font-size: 48px;

    line-height: 1.08;

    letter-spacing: -1.5px;

    margin: 0;

    color: #182230;
}

.hero p {
    max-width: 620px;

    margin: 15px auto 0;

    color: #687385;

    font-size: 15px;

    line-height: 1.65;
}


/* ---------------------------------------------------------
   QUESTION LABEL
--------------------------------------------------------- */

.question-label {
    font-size: 15px;

    font-weight: 600;

    color: #182230;

    text-align: left;

    margin-bottom: 8px;
}


/* ---------------------------------------------------------
   QUESTION TEXTAREA
--------------------------------------------------------- */

div[data-testid="stTextArea"] textarea {

    border: 1px solid #dfe4ea;

    border-radius: 12px;

    background: #fbfcfd;

    color: #182230;

    font-size: 14px;

    padding: 14px;

    min-height: 110px;
}

div[data-testid="stTextArea"] textarea:focus {

    border-color: #b08d57;

    box-shadow: 0 0 0 1px #b08d57;
}


/* ---------------------------------------------------------
   BUTTON
--------------------------------------------------------- */

.stButton > button {

    border-radius: 10px;

    font-weight: 600;

    min-height: 42px;
}

.stButton > button[kind="primary"] {

    background: #182230;

    border-color: #182230;

    color: #ffffff;
}

.stButton > button[kind="primary"]:hover {

    background: #263447;

    border-color: #263447;
}


/* ---------------------------------------------------------
   CHAT
--------------------------------------------------------- */

div[data-testid="stChatMessage"] {

    background: #ffffff;

    border: 1px solid #e3e7ec;

    border-radius: 14px;

    margin-bottom: 10px;
}


/* ---------------------------------------------------------
   DISCLAIMER
--------------------------------------------------------- */

.disclaimer {

    border-top: 1px solid #e3e7ec;

    margin-top: 35px;

    padding-top: 13px;

    color: #8b94a1;

    font-size: 11px;

    text-align: center;
}


/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media(max-width:700px) {

    .block-container {
        padding: 25px 18px 70px;
    }

    .hero {
        padding-top: 22px;
    }

    .hero h1 {
        font-size: 37px;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # -----------------------------
    # BRAND
    # -----------------------------

    st.markdown(
        """
        <div class="brand">

            <div class="brand-mark">
                ⚖
            </div>

            <div class="brand-name">
                LegalEase AI
            </div>

            <div class="brand-sub">
                Plain-language legal information
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # -----------------------------
    # AI CONNECTION
    # -----------------------------

    st.markdown(
        '<div class="sidebar-label">AI Connection</div>',
        unsafe_allow_html=True,
    )


    api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
    )


    models = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ]


    model = st.selectbox(
        "Model",
        models,
        index=models.index(
            st.session_state.model
        ),
    )


    # -----------------------------
    # CONNECT BUTTON
    # -----------------------------

    if st.button(
        "Connect AI",
        type="primary",
        use_container_width=True,
    ):

        # Reset previous state
        st.session_state.connected = False
        st.session_state.connection_error = ""


        # Empty key
        if not api_key.strip():

            st.session_state.connection_error = (
                "Please enter your OpenAI API key."
            )


        else:

            try:

                # Create OpenAI client
                client = OpenAI(
                    api_key=api_key.strip()
                )


                # Validate API key + selected model
                client.models.retrieve(model)


                # Save only after successful validation
                st.session_state.api_key = api_key.strip()

                st.session_state.model = model

                st.session_state.connected = True


            except AuthenticationError:

                st.session_state.connection_error = (
                    "Invalid OpenAI API key. "
                    "Please check your key and try again."
                )


            except RateLimitError:

                st.session_state.connection_error = (
                    "OpenAI rejected the connection because "
                    "of a rate-limit or billing/account restriction."
                )


            except APIConnectionError:

                st.session_state.connection_error = (
                    "Could not connect to OpenAI. "
                    "Check your connection and try again."
                )


            except APIError as e:

                st.session_state.connection_error = (
                    f"OpenAI API error: {e}"
                )


            except Exception as e:

                st.session_state.connection_error = (
                    f"Connection failed: {e}"
                )


    # -----------------------------
    # CONNECTION ERROR
    # -----------------------------

    if st.session_state.connection_error:

        st.markdown(
            f"""
            <div class="api-error">

                <strong>Connection failed</strong>

                <br>

                {st.session_state.connection_error}

            </div>
            """,
            unsafe_allow_html=True,
        )


    # -----------------------------
    # SUCCESS
    # -----------------------------

    if st.session_state.connected:

        st.markdown(
            """
            <div class="connection">

                <span class="dot"></span>

                OpenAI connected successfully

            </div>
            """,
            unsafe_allow_html=True,
        )


    st.divider()


    # =====================================================
    # LEGAL ASSISTANT
    # =====================================================

    st.markdown(
        '<div class="sidebar-label">Legal Assistant</div>',
        unsafe_allow_html=True,
    )


    focus_options = [
        "General Legal Information",
        "Contract Explanation",
        "Legal Document Summary",
        "Legal Terminology",
        "Basic Rights",
        "Legal Procedure",
        "Questions for a Lawyer",
        "Document Classification",
    ]


    st.session_state.focus = st.selectbox(
        "Focus",
        focus_options,
        index=focus_options.index(
            st.session_state.focus
        ),
    )


    st.divider()


    # =====================================================
    # CONVERSATION
    # =====================================================

    st.markdown(
        '<div class="sidebar-label">Conversation</div>',
        unsafe_allow_html=True,
    )


    if st.button(
        "＋ New conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


    st.markdown("---")

    st.caption("LegalEase AI")

    st.caption(
        "General information only • Not legal advice"
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-kicker">
            LEGAL INFORMATION, SIMPLIFIED
        </div>

        <h1>
            Understand the law.<br>
            Without the jargon.
        </h1>

        <p>
            Ask questions about legal terms, contracts,
            procedures and basic rights — and get clear
            explanations in everyday language.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# QUESTION AREA
# =========================================================

# Same container for label + textarea
# This makes their left edges align.

question_col = st.columns(
    [0.5, 9, 0.5]
)[1]


with question_col:

    st.markdown(
        '<div class="question-label">Ask LegalEase</div>',
        unsafe_allow_html=True,
    )


    question = st.text_area(
        "Your legal question",

        placeholder=(
            "Example: What can my employer "
            "terminate me for without notice?"
        ),

        height=110,

        label_visibility="collapsed",
    )


# =========================================================
# ASK BUTTON
# =========================================================

# IMPORTANT:
# This was missing in your previous code.
# That caused:
#
# NameError: name 'ask' is not defined

button_col = st.columns(
    [3, 1.5, 3]
)[1]


with button_col:

    ask = st.button(
        "⚖️  Ask LegalEase",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# SHOW PREVIOUS CHAT
# =========================================================

for msg in st.session_state.messages:

    with st.chat_message(
        "user"
        if isinstance(msg, HumanMessage)
        else "assistant"
    ):

        st.write(
            msg.content
        )


# =========================================================
# LEGAL QUESTION CLASSIFIER
# =========================================================

def is_legal_question(text):

    classifier = ChatOpenAI(
        model=st.session_state.model,
        temperature=0,
        api_key=st.session_state.api_key,
    )


    result = classifier.invoke(
        [

            SystemMessage(
                content="""
Classify the user's question.

Return ONLY one word:

LEGAL

or

NONLEGAL

LEGAL means the question substantially concerns:
- Law
- Legal rights
- Contracts
- Legal documents
- Legal procedures
- Courts
- Disputes
- Legal terminology
- Lawyers
- Legal preparation
- Legal obligations
- Legal responsibilities
- Legal claims

NONLEGAL means it is unrelated to law.
"""
            ),

            HumanMessage(
                content=text
            ),

        ]
    )


    return (
        result.content
        .strip()
        .upper()
        .startswith("LEGAL")
    )


# =========================================================
# PROCESS QUESTION
# =========================================================

if ask:

    # -----------------------------------------------------
    # 1. CHECK CONNECTION
    # -----------------------------------------------------

    if not st.session_state.connected:

        st.error(
            "Please connect a valid OpenAI API key "
            "before asking a question."
        )

        st.stop()


    # -----------------------------------------------------
    # 2. CHECK EMPTY QUESTION
    # -----------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a legal question."
        )

        st.stop()


    # -----------------------------------------------------
    # 3. CLASSIFY QUESTION
    # -----------------------------------------------------

    try:

        legal = is_legal_question(
            question.strip()
        )


    except AuthenticationError:

        st.error(
            "Your OpenAI API key is no longer valid. "
            "Please reconnect."
        )

        st.session_state.connected = False

        st.stop()


    except RateLimitError:

        st.error(
            "OpenAI returned a rate-limit or billing error. "
            "Please check your OpenAI account."
        )

        st.stop()


    except APIConnectionError:

        st.error(
            "Could not connect to OpenAI. "
            "Please try again."
        )

        st.stop()


    except APIError as e:

        st.error(
            f"OpenAI API error: {e}"
        )

        st.stop()


    except Exception as e:

        st.error(
            f"Unable to classify the question: {e}"
        )

        st.stop()


    # -----------------------------------------------------
    # 4. DISPLAY USER QUESTION
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.write(
            question.strip()
        )


    # -----------------------------------------------------
    # 5. REJECT NON-LEGAL QUESTION
    # -----------------------------------------------------

    if not legal:

        with st.chat_message("assistant"):

            st.warning(
                NONLEGAL
            )

        st.stop()


    # -----------------------------------------------------
    # 6. SAVE LEGAL QUESTION
    # -----------------------------------------------------

    st.session_state.messages.append(
        HumanMessage(
            content=question.strip()
        )
    )


    # -----------------------------------------------------
    # 7. CREATE LEGAL AI
    # -----------------------------------------------------

    try:

        chat = ChatOpenAI(
            model=st.session_state.model,
            temperature=0.2,
            api_key=st.session_state.api_key,
        )


        conversation = [

            SystemMessage(
                content=(
                    LEGAL_PROMPT
                    + "\n\nCurrent focus: "
                    + st.session_state.focus
                )
            )

        ] + st.session_state.messages


        # -------------------------------------------------
        # 8. GENERATE RESPONSE
        # -------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Preparing a clear legal explanation..."
            ):

                response = chat.invoke(
                    conversation
                )


            st.write(
                response.content
            )


        # -------------------------------------------------
        # 9. SAVE AI RESPONSE
        # -------------------------------------------------

        st.session_state.messages.append(
            AIMessage(
                content=response.content
            )
        )


    # -----------------------------------------------------
    # API ERRORS
    # -----------------------------------------------------

    except AuthenticationError:

        st.error(
            "Your OpenAI API key is no longer valid. "
            "Please reconnect."
        )

        st.session_state.connected = False


    except RateLimitError:

        st.error(
            "OpenAI returned a rate-limit or billing error. "
            "Please check your OpenAI account."
        )


    except APIConnectionError:

        st.error(
            "Could not connect to OpenAI. "
            "Please try again."
        )


    except APIError as e:

        st.error(
            f"OpenAI API error: {e}"
        )


    except Exception as e:

        st.error(
            f"Something went wrong: {e}"
        )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown(
    """
    <div class="disclaimer">
        LegalEase AI provides general legal information
        and does not replace advice from a qualified
        legal professional.
    </div>
    """,
    unsafe_allow_html=True,
)
