"""Admin page: add businesses, upload their documents and data, view tickets.

Streamlit shows every file in the pages/ folder as an extra page in the sidebar.
Set ADMIN_PASSWORD in .env to protect this page.
"""

import json

import streamlit as st

from src import config, db, rag

st.set_page_config(page_title="Admin · AI Support Assistant", page_icon="⚙️", layout="wide")

CUSTOMERS_EXAMPLE = """[
  {"customer_id": "CUST001", "name": "Asha Rao", "email": "asha@example.com",
   "phone": "+91-90000-00001", "city": "Pune", "membership": "Gold", "member_since": "2025-01-10"}
]"""

ORDERS_EXAMPLE = """[
  {"order_id": "ORD1001", "customer_id": "CUST001", "product": "Chocolate Cake",
   "amount": 650, "order_date": "2026-10-20", "status": "Processing",
   "estimated_delivery": "2026-10-22", "tracking_status": "Being prepared"}
]"""

db.init_db()

# ---------------------------------------------------------------------------
# Optional password
# ---------------------------------------------------------------------------
if config.ADMIN_PASSWORD and not st.session_state.get("admin_ok"):
    st.title("⚙️ Admin")
    password = st.text_input("Admin password", type="password")
    if password:
        if password == config.ADMIN_PASSWORD:
            st.session_state.admin_ok = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()

st.title("⚙️ Admin")
st.caption("Each business has its own settings, knowledge base, customers, orders and tickets. "
           "Nothing is shared between businesses.")


def read_upload(uploaded) -> str:
    """Decode an uploaded text file. utf-8-sig also handles files saved by Notepad."""
    return uploaded.getvalue().decode("utf-8-sig")


# ---------------------------------------------------------------------------
# Add a business
# ---------------------------------------------------------------------------
with st.expander("➕ Add a business", expanded=not db.list_businesses()):
    with st.form("new_business", clear_on_submit=True):
        col1, col2 = st.columns(2)
        new_id = col1.text_input("Business ID", placeholder="city-bakery",
                                 help="Lowercase letters, numbers and dashes. Used in the chat link.")
        new_name = col2.text_input("Business name", placeholder="City Bakery")
        new_description = st.text_input(
            "What the business does (one line)",
            placeholder="a bakery in Pune that delivers cakes and bread",
            help="Completes the sentence '<name> is ...'. The assistant uses it to understand questions.",
        )
        col3, col4, col5 = st.columns(3)
        new_logo = col3.text_input("Logo emoji", value="💬", max_chars=4)
        new_color = col4.color_picker("Accent colour", value="#4f46e5")
        new_currency = col5.text_input("Currency", value="INR", max_chars=5)
        new_key = st.text_input("OpenAI API key (optional)", type="password",
                                help="Leave empty to use OPENAI_API_KEY from .env.")
        if st.form_submit_button("Create business", type="primary"):
            try:
                created = db.create_business(
                    new_id, new_name, description=new_description, logo=new_logo,
                    accent_color=new_color, currency=new_currency, openai_api_key=new_key,
                )
                st.session_state.admin_business = created["business_id"]
                st.success(f"Created {created['name']}. Now upload its documents below.")
            except db.DataError as exc:
                st.error(str(exc))

businesses = db.list_businesses()
if not businesses:
    st.info("No businesses yet. Add one above.")
    st.stop()

# ---------------------------------------------------------------------------
# Manage one business
# ---------------------------------------------------------------------------
ids = [b["business_id"] for b in businesses]
current = st.session_state.get("admin_business")
business_id = st.selectbox(
    "Manage business",
    ids,
    index=ids.index(current) if current in ids else 0,
    format_func=lambda bid: next(f"{b['logo']} {b['name']} ({bid})" for b in businesses
                                 if b["business_id"] == bid),
)
st.session_state.admin_business = business_id
business = db.get_business(business_id)
st.markdown(f"Chat link for this business: `http://localhost:8501/?business={business_id}`")

tab_kb, tab_data, tab_tickets, tab_settings = st.tabs(
    ["📚 Knowledge base", "👥 Customers & orders", "🎫 Tickets", "🛠️ Settings"]
)

# ---------- Knowledge base ----------
with tab_kb:
    documents = db.list_documents(business_id)
    info = rag.index_info(business_id)
    if info:
        st.caption(f"Search index: {info['documents']} documents, {info['chunks']} chunks, "
                   f"built with `{info['embedding_model']}`.")
    else:
        st.caption("Search index: not built yet. It is built automatically on the first "
                   "question, or with the button below.")

    if documents:
        for doc in documents:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"**{doc['filename']}** · {len(doc['content']):,} characters · "
                          f"uploaded {doc['uploaded_at']}")
            if col2.button("Delete", key=f"del_{doc['filename']}"):
                db.delete_document(business_id, doc["filename"])
                rag.delete_vector_store(business_id)  # rebuilt on the next question
                st.rerun()
    else:
        st.info("No documents yet. Upload policy files, FAQs or help articles as .txt or .md.")

    uploads = st.file_uploader(
        "Upload documents (.txt or .md). A file with the same name replaces the old one.",
        type=["txt", "md"],
        accept_multiple_files=True,
        key=f"docs_{business_id}",
    )
    col1, col2 = st.columns(2)
    if col1.button("Save uploaded documents", type="primary", disabled=not uploads):
        try:
            for uploaded in uploads:
                db.add_document(business_id, uploaded.name, read_upload(uploaded))
            rag.delete_vector_store(business_id)
            st.success(f"Saved {len(uploads)} document(s). The search index will be rebuilt.")
        except (db.DataError, UnicodeDecodeError) as exc:
            st.error(f"Could not save: {exc}")
    if col2.button("Rebuild search index now", disabled=not documents):
        try:
            with st.spinner("Building the index..."):
                rag.build_vector_store(business_id)
            st.success("Search index rebuilt.")
            st.rerun()
        except rag.KnowledgeBaseError as exc:
            st.error(str(exc))

# ---------- Customers & orders ----------
with tab_data:
    st.caption("In a real product these would come from the business's own systems "
               "(Shopify, a CRM, an order API). Here you upload them as JSON files.")
    customers = db.list_customers(business_id)
    orders = db.list_orders(business_id)
    st.markdown(f"**Customers ({len(customers)})**")
    if customers:
        st.dataframe([{k: v for k, v in c.items() if k != "business_id"} for c in customers],
                     hide_index=True, width="stretch")
    st.markdown(f"**Orders ({len(orders)})**")
    if orders:
        st.dataframe([{k: v for k, v in o.items() if k != "business_id"} for o in orders],
                     hide_index=True, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader("Replace customers (customers.json)", type=["json"],
                                key=f"customers_{business_id}")
        with st.popover("File format"):
            st.code(CUSTOMERS_EXAMPLE, language="json")
            st.caption("Orders of customers missing from the new file are removed.")
        if file and st.button("Import customers", type="primary"):
            try:
                count = db.replace_customers(business_id, json.loads(read_upload(file)))
                st.success(f"Imported {count} customers.")
                st.rerun()
            except (db.DataError, ValueError) as exc:
                st.error(f"Could not import: {exc}")
    with col2:
        file = st.file_uploader("Replace orders (orders.json)", type=["json"],
                                key=f"orders_{business_id}")
        with st.popover("File format"):
            st.code(ORDERS_EXAMPLE, language="json")
            st.caption("Every customer_id must already exist. Upload customers first.")
        if file and st.button("Import orders", type="primary"):
            try:
                count = db.replace_orders(business_id, json.loads(read_upload(file)))
                st.success(f"Imported {count} orders.")
                st.rerun()
            except (db.DataError, ValueError) as exc:
                st.error(f"Could not import: {exc}")

# ---------- Tickets ----------
with tab_tickets:
    tickets = db.list_tickets(business_id)
    if tickets:
        st.dataframe([{k: v for k, v in t.items() if k != "business_id"} for t in tickets],
                     hide_index=True, width="stretch")
    else:
        st.info("No tickets yet. The assistant creates them when a customer needs a human agent.")

# ---------- Settings ----------
with tab_settings:
    with st.form(f"settings_{business_id}"):
        name = st.text_input("Business name", value=business["name"])
        description = st.text_input("What the business does (one line)",
                                    value=business["description"])
        col1, col2, col3 = st.columns(3)
        logo = col1.text_input("Logo emoji", value=business["logo"], max_chars=4)
        color = col2.color_picker("Accent colour", value=business["accent_color"])
        currency = col3.text_input("Currency", value=business["currency"], max_chars=5)
        if business["has_own_api_key"]:
            st.caption("🔑 This business uses its own OpenAI API key.")
        else:
            st.caption("This business uses OPENAI_API_KEY from .env.")
        new_key = st.text_input("New OpenAI API key (leave empty to keep the current setting)",
                                type="password")
        remove_key = st.checkbox("Remove this business's own key and use the .env key",
                                 disabled=not business["has_own_api_key"])
        if st.form_submit_button("Save settings", type="primary"):
            changes = dict(name=name, description=description, logo=logo,
                           accent_color=color, currency=currency)
            if remove_key:
                changes["openai_api_key"] = ""
            elif new_key.strip():
                changes["openai_api_key"] = new_key
            try:
                db.update_business(business_id, **changes)
                st.success("Settings saved.")
                st.rerun()
            except db.DataError as exc:
                st.error(str(exc))

    st.divider()
    st.markdown("**Delete this business**")
    st.caption("Deletes its settings, documents, customers, orders, tickets and search index. "
               "This cannot be undone.")
    confirm = st.text_input(f"Type `{business_id}` to confirm", key=f"confirm_{business_id}")
    if st.button("Delete business", disabled=confirm.strip() != business_id):
        db.delete_business(business_id)
        rag.delete_vector_store(business_id)
        st.session_state.pop("admin_business", None)
        st.success(f"Deleted {business['name']}.")
        st.rerun()
