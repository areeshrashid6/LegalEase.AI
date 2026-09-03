import streamlit as st
from openai import OpenAI, AuthenticationError, APIConnectionError, APIError, RateLimitError
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="LegalEase AI", page_icon="⚖️", layout="wide")

LEGAL_PROMPT = """You are LegalEase AI, a domain-specific legal information assistant.
Answer ONLY questions substantially related to law or legal information.
Supported: legal terminology, contracts, legal documents, basic rights,
employment law, consumer rights, property/rental law, business law, family law,
intellectual property, legal procedures, courts/disputes, lawyer-question
preparation, and basic legal document classification.

If the question is not legal, reply exactly:
"I don't have training for that topic. LegalEase AI is designed specifically for
legal-information questions. Please ask me about legal terminology, contracts,
rights, legal procedures, or another legal topic."

Provide general legal information, not legal advice. Laws vary by jurisdiction.
Ask for country/state/province when jurisdiction matters. Never invent laws,
cases, citations, deadlines, penalties, or requirements. Explain simply."""

NONLEGAL = ("I don't have training for that topic. LegalEase AI is designed "
            "specifically for legal-information questions. Please ask me about "
            "legal terminology, contracts, rights, legal procedures, or another "
            "legal topic.")

defaults = {
    "messages": [], "api_key": "", "model": "gpt-4o-mini",
    "connected": False, "connection_error": "",
    "focus": "General Legal Information"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f7f8fa;color:#182230}
.block-container{max-width:1120px;padding:38px 46px 70px}
section[data-testid="stSidebar"]{background:#eef1f5;border-right:1px solid #dfe4ea}
section[data-testid="stSidebar"]>div{padding:28px 20px}
.brand{padding:4px 2px 24px}
.brand-mark{width:42px;height:42px;border-radius:12px;background:#182230;color:white;display:flex;align-items:center;justify-content:center;font-size:21px;margin-bottom:12px}
.brand-name{font-family:'Playfair Display',serif;font-size:25px;font-weight:700;color:#182230}
.brand-sub{color:#77808d;font-size:11px;margin-top:3px}
.sidebar-label{color:#8a94a2;font-size:10px;font-weight:700;letter-spacing:1.3px;text-transform:uppercase;margin:16px 0 8px}
.connection{background:#f7fcf8;border:1px solid #d5e8dc;border-radius:11px;padding:9px 11px;color:#52605a;font-size:12px;margin-top:10px}
.dot{display:inline-block;width:7px;height:7px;background:#42a36a;border-radius:50%;margin-right:7px}
.api-error{background:#fff6f5;border:1px solid #f0c8c3;color:#9b332a;border-radius:11px;padding:11px;margin-top:10px;font-size:12px;line-height:1.5}
.hero {
    text-align: center;
    padding: 55px 20px 25px;
}
.hero-kicker{color:#b08d57;font-size:10px;font-weight:700;letter-spacing:1.8px;margin-bottom:13px}
.hero h1{font-family:'Playfair Display',serif;font-size:48px;line-height:1.08;letter-spacing:-1.5px;margin:0;color:#182230}
.hero p{max-width:620px;margin:15px auto 0;color:#687385;font-size:15px;line-height:1.65}
.question-area{max-width:820px;margin:18px auto 26px;background:#fff;border:1px solid #e3e7ec;border-radius:18px;padding:18px;box-shadow:0 10px 35px rgba(20,30,45,.05)}
.question-label {
    font-size: 15px;
    font-weight: 600;
    color: #182230;
    text-align: left;
    margin-bottom: 8px;
}
div[data-testid="stTextArea"] textarea{border:1px solid #dfe4ea;border-radius:12px;background:#fbfcfd;color:#182230;font-size:14px;padding:14px;min-height:110px}
div[data-testid="stTextArea"] textarea:focus{border-color:#b08d57;box-shadow:0 0 0 1px #b08d57}
.stButton>button{border-radius:10px;font-weight:600;min-height:42px}
.stButton>button[kind="primary"]{background:#182230;border-color:#182230;color:#fff}
.stButton>button[kind="primary"]:hover{background:#263447;border-color:#263447}
.intro{max-width:820px;margin:0 auto 22px;background:#fff;border:1px solid #e3e7ec;border-radius:16px;padding:18px 20px}
.intro-title{font-weight:700;font-size:15px;margin-bottom:4px}.intro-text{color:#687385;font-size:13px;line-height:1.55}
.feature{background:#fff;border:1px solid #e3e7ec;border-radius:15px;padding:17px;min-height:126px;margin-bottom:14px}
.feature-icon{font-size:21px;margin-bottom:9px}.feature-title{font-size:14px;font-weight:700;margin-bottom:5px}.feature-text{color:#687385;font-size:12px;line-height:1.5}
div[data-testid="stChatMessage"]{background:#fff;border:1px solid #e3e7ec;border-radius:14px;margin-bottom:10px}
.disclaimer{border-top:1px solid #e3e7ec;margin-top:26px;padding-top:13px;color:#8b94a1;font-size:11px;text-align:center}
@media(max-width:700px){.block-container{padding:25px 18px 70px}.hero{padding-top:22px}.hero h1{font-size:37px}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div class="brand"><div class="brand-mark">⚖</div>
<div class="brand-name">LegalEase AI</div><div class="brand-sub">Plain-language legal information</div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">AI Connection</div>', unsafe_allow_html=True)
    api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password", placeholder="sk-...")
    models = ["gpt-4o-mini","gpt-4o","gpt-4.1-mini","gpt-4.1"]
    model = st.selectbox("Model", models, index=models.index(st.session_state.model))
    if st.button("Connect AI", type="primary", use_container_width=True):
        st.session_state.connected = False
        st.session_state.connection_error = ""
        if not api_key.strip():
            st.session_state.connection_error = "Please enter your OpenAI API key."
        else:
            try:
                client = OpenAI(api_key=api_key.strip())
                client.models.retrieve(model)
                st.session_state.api_key = api_key.strip()
                st.session_state.model = model
                st.session_state.connected = True
            except AuthenticationError:
                st.session_state.connection_error = "Invalid OpenAI API key. Please check the key and try again."
            except RateLimitError:
                st.session_state.connection_error = "OpenAI rejected the connection because of a rate-limit or billing/account restriction."
            except APIConnectionError:
                st.session_state.connection_error = "Could not connect to OpenAI. Check your connection and try again."
            except APIError as e:
                st.session_state.connection_error = f"OpenAI API error: {e}"
            except Exception as e:
                st.session_state.connection_error = f"Connection failed: {e}"
    if st.session_state.connection_error:
        st.markdown(f'<div class="api-error"><strong>Connection failed</strong><br>{st.session_state.connection_error}</div>', unsafe_allow_html=True)
    if st.session_state.connected:
        st.markdown('<div class="connection"><span class="dot"></span>OpenAI connected successfully</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-label">Legal Assistant</div>', unsafe_allow_html=True)
    focus_options = ["General Legal Information","Contract Explanation","Legal Document Summary","Legal Terminology","Basic Rights","Legal Procedure","Questions for a Lawyer","Document Classification"]
    st.session_state.focus = st.selectbox("Focus", focus_options, index=focus_options.index(st.session_state.focus))
    st.divider()
    st.markdown('<div class="sidebar-label">Conversation</div>', unsafe_allow_html=True)
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("LegalEase AI")
    st.caption("General information only • Not legal advice")

st.markdown("""<div class="hero"><div class="hero-kicker">LEGAL INFORMATION, SIMPLIFIED</div>
<h1>Understand the law.<br>Without the jargon.</h1>
<p>Ask questions about legal terms, contracts, procedures and basic rights — and get clear explanations in everyday language.</p></div>""", unsafe_allow_html=True)

# ==============================
# ASK LEGALEASE
# ==============================

question_col = st.columns([0.5, 9, 0.5])[1]

with question_col:

    st.markdown(
        '<div class="question-label">Ask LegalEase</div>',
        unsafe_allow_html=True
    )

    question = st.text_area(
        "Your legal question",
        placeholder="Example: What can my employer terminate me for without notice?",
        height=110,
        label_visibility="collapsed"
    )


# ==============================
# ASK BUTTON
# ==============================

button_col = st.columns([3, 1.5, 3])[1]

with button_col:

    ask = st.button(
        "⚖️  Ask LegalEase",
        type="primary",
        use_container_width=True
    )


def is_legal_question(text):
    classifier=ChatOpenAI(model=st.session_state.model,temperature=0,api_key=st.session_state.api_key)
    result=classifier.invoke([
        SystemMessage(content="Classify the question. Return ONLY LEGAL or NONLEGAL. LEGAL means it substantially concerns law, legal rights, contracts, legal documents, legal procedures, courts, disputes, legal terminology, or preparation for legal counsel."),
        HumanMessage(content=text)
    ])
    return result.content.strip().upper().startswith("LEGAL")

if ask:
    if not st.session_state.connected:
        st.error("Please connect a valid OpenAI API key before asking a question.")
        st.stop()
    if not question.strip():
        st.warning("Please enter a legal question.")
        st.stop()
    try:
        legal=is_legal_question(question.strip())
    except AuthenticationError:
        st.error("Your OpenAI API key is no longer valid. Please reconnect.")
        st.session_state.connected=False
        st.stop()
    except Exception as e:
        st.error(f"Unable to classify the question: {e}")
        st.stop()

    with st.chat_message("user"):
        st.write(question.strip())
    if not legal:
        with st.chat_message("assistant"):
            st.warning(NONLEGAL)
        st.stop()

    st.session_state.messages.append(HumanMessage(content=question.strip()))
    try:
        chat=ChatOpenAI(model=st.session_state.model,temperature=0.2,api_key=st.session_state.api_key)
        conversation=[SystemMessage(content=LEGAL_PROMPT+f"\n\nCurrent focus: {st.session_state.focus}")] + st.session_state.messages
        with st.chat_message("assistant"):
            with st.spinner("Preparing a clear legal explanation..."):
                response=chat.invoke(conversation)
            st.write(response.content)
        st.session_state.messages.append(AIMessage(content=response.content))
    except AuthenticationError:
        st.error("Your OpenAI API key is no longer valid. Please reconnect.")
        st.session_state.connected=False
    except RateLimitError:
        st.error("OpenAI returned a rate-limit or billing error. Please check your account.")
    except APIConnectionError:
        st.error("Could not connect to OpenAI. Please try again.")
    except APIError as e:
        st.error(f"OpenAI API error: {e}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown('<div class="disclaimer">LegalEase AI provides general legal information and does not replace advice from a qualified legal professional.</div>', unsafe_allow_html=True)
