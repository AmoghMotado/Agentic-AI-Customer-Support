"""Streamlit chat UI: one support assistant that serves many businesses.

Run with:
    python -m streamlit run app.py

Pick a business in the sidebar, or open a business's own link, for example
http://localhost:8501/?business=greenleaf
The Admin page (pages/1_Admin.py) adds businesses and uploads their documents.
"""

import html
import logging

import streamlit as st

from src import config, db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="AI Support Assistant", page_icon="💬", layout="centered")

ACTION_LABELS = {
    "rag": "📚 Knowledge base (RAG)",
    "tool": "🛠️ Tool call",
    "direct": "💬 Direct response",
    "clarify": "❓ Clarification",
    "error": "⚠️ Error",
    "none": "-",
}


@st.cache_resource
def setup_database():
    """Create the database and seed the demo businesses (runs once per app start)."""
    db.init_db()
    return True


@st.cache_resource
def load_graph():
    from src.graph import run_turn

    return run_turn


def load_knowledge_base(business_id: str):
    """Load (or build) this business's FAISS index. It is cached inside src.rag."""
    from src.rag import get_vector_store

    return get_vector_store(business_id)


def example_queries(business_id: str) -> list[str]:
    """Example buttons that use real order IDs from this business's data."""
    orders = db.list_orders(business_id)
    examples = ["What is your refund policy?", "How long does delivery take?"]
    if orders:
        examples.append(f"Where is order {orders[0]['order_id']}?")
    examples += ["Where is my order?", "My order arrived damaged. Create a support ticket."]
    processing = next((o for o in orders if o["status"] == "Processing"), None)
    if processing:
        examples.append(f"Can I cancel order {processing['order_id']}?")
    examples += ["What payment methods do you support?", "Hello"]
    return examples


def render_debug(debug: dict):
    """Show SAFE execution metadata only (no hidden reasoning)."""
    if not debug:
        return
    with st.expander("Execution details", expanded=False):
        st.markdown(f"**Action:** {ACTION_LABELS.get(debug.get('action'), debug.get('action'))}")
        if debug.get("business_id"):
            st.markdown(f"**Business:** `{debug['business_id']}`")
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


setup_database()
businesses = db.list_businesses()
if not businesses:
    st.error("No businesses yet. Add one on the **Admin** page.")
    st.stop()

# ---------------------------------------------------------------------------
# Which business is this chat for?
# ---------------------------------------------------------------------------
# A link like ?business=greenleaf picks the business; otherwise the default one.
business_ids = [b["business_id"] for b in businesses]
requested = st.query_params.get("business", config.DEFAULT_BUSINESS_ID)
if requested not in business_ids:
    requested = business_ids[0]

with st.sidebar:
    business_id = st.selectbox(
        "Business",
        business_ids,
        index=business_ids.index(requested),
        format_func=lambda bid: next(f"{b['logo']} {b['name']}" for b in businesses
                                     if b["business_id"] == bid),
        help="Each business has its own documents, customers, orders and tickets.",
    )
st.query_params["business"] = business_id
business = next(b for b in businesses if b["business_id"] == business_id)

# A conversation belongs to one business. Switching business starts a new chat,
# so one business's chat history is never sent to the LLM as another's.
if st.session_state.get("business_id") != business_id:
    st.session_state.business_id = business_id
    st.session_state.messages = []  # [{"role", "content", "debug"}]
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

accent = html.escape(business["accent_color"])
st.markdown(
    f"""
    <style>
      .app-subtitle {{ color: #6b7280; margin-top: -0.75rem; margin-bottom: 1rem; }}
      .badge {{ display:inline-block; padding:2px 10px; border-radius:999px;
               border:1px solid {accent}; color:{accent}; font-size:0.8rem; margin-right:6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    customers = db.list_customers(business_id)
    demo_ids = ", ".join(c["customer_id"] for c in customers) or "none yet"
    customer_id = st.text_input(
        "Customer ID",
        value=customers[0]["customer_id"] if customers else "",
        key=f"customer_id_{business_id}",  # a separate box per business
        help=f"Customers of this business: {demo_ids}. Leave empty to test the missing-ID flow.",
    ).strip().upper()

    if customer_id:
        customer = db.get_customer(business_id, customer_id)
        if customer:
            membership = f" ({customer['membership']})" if customer["membership"] else ""
            st.caption(f"Signed in as **{customer['name']}**{membership}")
        else:
            st.warning(f"Unknown customer ID for {business['name']}: {customer_id}")
    else:
        st.caption("No customer ID set.")

    st.subheader("Example queries")
    for query in example_queries(business_id):
        if st.button(query, width="stretch", key=f"example_{query}"):
            st.session_state.pending_query = query

    st.divider()
    debug_mode = st.toggle("Debug mode", value=False, help="Show the route, tools and sources used.")

    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Demo data"):
        st.caption(f"Delete {business['name']}'s tickets and restore its demo orders.")
        if st.button("Reset demo data", width="stretch"):
            db.reset_demo_data(business_id)
            st.success("Demo data reset.")
        st.caption(f"Model: `{config.OPENAI_MODEL}`  \nEmbeddings: `{config.EMBEDDING_PROVIDER}`")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title(f"{business['logo']} {business['name']} Support")
st.markdown(
    '<p class="app-subtitle">AI-powered customer support using RAG, agents and tool calling.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="badge">LangGraph</span><span class="badge">RAG + FAISS</span>'
    '<span class="badge">Tool calling</span><span class="badge">Multi-business</span>',
    unsafe_allow_html=True,
)
st.write("")

# Startup checks: show clear messages instead of stack traces.
from src.agent import api_key_for  # noqa: E402  (imported here to keep startup fast)

ready = True
if not config.key_looks_real(api_key_for(business_id)):
    ready = False
    st.error(
        "**OpenAI API key not found.** Copy `.env.example` to `.env`, put your key in "
        "`OPENAI_API_KEY`, then restart the app with `python -m streamlit run app.py`. "
        "(A business can also have its own key, set on the Admin page.)"
    )
else:
    try:
        with st.spinner(f"Loading {business['name']}'s knowledge base "
                        "(the first run downloads the embedding model)..."):
            load_knowledge_base(business_id)
    except Exception as exc:  # KnowledgeBaseError or embedding download problems
        ready = False
        logging.exception("Knowledge base failed to load")
        st.error(f"**Knowledge base could not be loaded.** {exc}")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            f"Hi! I'm {business['name']}'s support assistant. I can answer questions about "
            "our policies, check your order status, cancel orders and create support tickets. "
            "How can I help?"
        )

# Replay the conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if debug_mode and msg["role"] == "assistant":
            render_debug(msg.get("debug"))

typed = st.chat_input("Ask about orders, refunds, delivery...", disabled=not ready)
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
                result = run_turn(user_input, history, business_id, customer_id)
            if result.get("action") == "error":
                st.error(result["response"])
            else:
                st.markdown(result["response"])
            if debug_mode:
                render_debug(result)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["response"], "debug": result}
        )
