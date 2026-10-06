"""Streamlit UI for the NovaCart Agentic AI Customer Support Assistant.

Run with:
    streamlit run app.py
"""

import json
import logging

import streamlit as st

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="NovaCart Support Assistant",
    page_icon="🛒",
    layout="centered",
)

EXAMPLE_QUERIES = [
    "What is your refund policy?",
    "Can I return a damaged product?",
    "Where is order ORD1001?",
    "Where is my order?",
    "My package arrived damaged. Create a support ticket.",
    "Can I cancel order ORD1003?",
    "What payment methods do you support?",
    "Hello",
]

ACTION_LABELS = {
    "rag": "📚 Knowledge base (RAG)",
    "tool": "🛠️ Tool call",
    "direct": "💬 Direct response",
    "clarify": "❓ Clarification",
    "error": "⚠️ Error",
    "none": "-",
}

st.markdown(
    """
    <style>
      .app-subtitle { color: #6b7280; margin-top: -0.75rem; margin-bottom: 1rem; }
      .badge { display:inline-block; padding:2px 10px; border-radius:999px;
               background:#eef2ff; color:#3730a3; font-size:0.8rem; margin-right:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "debug"}]
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


@st.cache_resource(show_spinner="Loading the knowledge base (first run downloads the embedding model)...")
def load_knowledge_base():
    """Load (or build) the FAISS index once per app session."""
    from src.rag import get_vector_store

    return get_vector_store()


@st.cache_resource
def load_graph():
    from src.graph import run_turn

    return run_turn


def load_customer(customer_id: str):
    try:
        with open(config.CUSTOMERS_FILE, encoding="utf-8") as f:
            customers = json.load(f)
    except (OSError, ValueError):
        return None
    return next((c for c in customers if c["customer_id"] == customer_id), None)


def render_debug(debug: dict):
    """Show SAFE execution metadata only (no hidden reasoning)."""
    if not debug:
        return
    with st.expander("Execution details", expanded=False):
        st.markdown(f"**Action:** {ACTION_LABELS.get(debug.get('action'), debug.get('action'))}")
        if debug.get("route_label"):
            st.markdown(f"**Request type:** {debug['route_label']}")
        if debug.get("search_query"):
            st.markdown(f"**Search query:** {debug['search_query']}")
        if debug.get("sources"):
            st.markdown("**Retrieved sources** (lower distance = more relevant):")
            for s in debug["sources"]:
                st.markdown(f"- `{s['source']}` chunk {s['chunk']} (distance {s['score']})")
        if debug.get("missing_info"):
            st.markdown(f"**Missing information:** {debug['missing_info']}")
        for call in debug.get("tool_calls", []):
            st.markdown(f"**Tool:** `{call['tool']}`")
            st.markdown("**Tool input:**")
            st.json(call["input"])
            st.markdown("**Tool result:**")
            st.json(call["result"])
        if debug.get("error"):
            st.markdown(f"**Error type:** {debug['error']}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🛒 NovaCart")
    customer_id = st.text_input(
        "Customer ID",
        value="CUST001",
        help="Demo customers: CUST001, CUST002, CUST003. Leave empty to test the missing-ID flow.",
    ).strip().upper()

    if customer_id:
        customer = load_customer(customer_id)
        if customer:
            st.caption(f"Signed in as **{customer['name']}** ({customer['membership']})")
        else:
            st.warning(f"Unknown customer ID: {customer_id}")
    else:
        st.caption("No customer ID set.")

    st.subheader("Example queries")
    for query in EXAMPLE_QUERIES:
        if st.button(query, use_container_width=True):
            st.session_state.pending_query = query

    st.divider()
    debug_mode = st.toggle("Debug mode", value=False, help="Show the route, tools and sources used.")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Demo data"):
        st.caption("Restore orders and delete all created tickets.")
        if st.button("Reset demo data", use_container_width=True):
            from src.tools import reset_demo_data

            reset_demo_data()
            st.success("Demo data reset.")
        st.caption(f"Model: `{config.OPENAI_MODEL}`  \nEmbeddings: `{config.EMBEDDING_PROVIDER}`")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Agentic AI Customer Support Assistant")
st.markdown(
    '<p class="app-subtitle">AI-powered customer support using RAG, agents and tool calling.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="badge">LangGraph</span><span class="badge">RAG + FAISS</span>'
    '<span class="badge">Tool calling</span><span class="badge">OpenAI</span>',
    unsafe_allow_html=True,
)
st.write("")

# Startup checks: show clear messages instead of stack traces.
ready = True
if not config.api_key_is_set():
    ready = False
    st.error(
        "**OpenAI API key not found.** Copy `.env.example` to `.env`, put your key in "
        "`OPENAI_API_KEY`, then restart the app with `streamlit run app.py`."
    )
else:
    try:
        load_knowledge_base()
    except Exception as exc:  # KnowledgeBaseError or embedding download problems
        ready = False
        logging.exception("Knowledge base failed to load")
        st.error(f"**Knowledge base could not be loaded.** {exc}")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi! I'm NovaCart's support assistant. I can answer questions about our "
            "policies, check your order status, cancel orders and create support tickets. "
            "How can I help?"
        )

# Replay the conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if debug_mode and msg["role"] == "assistant":
            render_debug(msg.get("debug"))

typed = st.chat_input("Ask about orders, refunds, shipping...", disabled=not ready)
user_input = typed or st.session_state.pending_query
st.session_state.pending_query = None

if user_input and ready:
    user_input = user_input.strip()
    if user_input:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                run_turn = load_graph()
                result = run_turn(user_input, history, customer_id)
            if result.get("action") == "error":
                st.error(result["response"])
            else:
                st.markdown(result["response"])
            if debug_mode:
                render_debug(result)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["response"], "debug": result}
        )
