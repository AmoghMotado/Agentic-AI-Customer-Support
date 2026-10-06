# Test Checklist

## 1. Automated tests (no API key needed, free)

```powershell
python -m pytest -v
```

Expected: `24 passed, 5 deselected`. The 5 deselected tests are the slow retrieval tests. Run them with:

```powershell
python -m pytest -v -m slow
```

Expected: `5 passed` (the first run downloads the embedding model, about 90 MB).

## 2. Manual tests in the app

Start the app with `streamlit run app.py`, turn on **Debug mode** in the sidebar, and keep Customer ID = `CUST001` unless a row says otherwise. Use **Clear conversation** between rows unless the row is a multi-turn test.

### RAG

| # | Message | Expected action | Expected answer contains |
|---|---|---|---|
| R1 | What is your refund policy? | RAG, source `refund_policy.txt` | 30 days, original payment method |
| R2 | How do returns work for electronics? | RAG, `return_policy.txt` | 10 days |
| R3 | How long does delivery take? | RAG, `shipping_policy.txt` | 3 to 7 business days |
| R4 | What payment methods do you support? | RAG, `payment_policy.txt` | UPI, cards, COD |
| R5 | Do you offer price matching? (unknown policy) | RAG | Says it couldn't find this in NovaCart's policies, offers a ticket. Must NOT invent a policy. |

### Tools

| # | Customer ID | Message | Expected |
|---|---|---|---|
| T1 | CUST001 | Where is order ORD1001? | Tool `get_order_status`, input ORD1001, answer: Shipped, 2026-10-07 |
| T2 | CUST001 | Where is order ORD9999? | Tool called, result `ok: false`, polite "order not found" |
| T3 | CUST001 | Where is my order? | Clarification asking for the order ID, **no tool call** |
| T4 | CUST001 | My package for ORD1002 arrived damaged. Create a support ticket. | Tool `create_support_ticket`, answer contains a ticket ID like TCK1001 |
| T5 | (empty) | My package arrived damaged. Create a support ticket. | Clarification asking for customer ID, **no ticket created** |
| T6 | CUST002 | Can I cancel order ORD1003? | Tool `cancel_order`, order cancelled |
| T7 | CUST001 | Cancel order ORD1001 | Tool result says it cannot be cancelled (Shipped) |
| T8 | CUST001 | Where is order ORD1003? | Tool result: not linked to this customer (it belongs to CUST002) |
| T9 | CUST001 | Show my account details | Tool `get_customer_details`, Aarav Sharma, 2 orders |

Use **Demo data → Reset demo data** in the sidebar after T6 to restore ORD1003.

### Agent routing

| # | Message | Expected action |
|---|---|---|
| A1 | Hello | Direct |
| A2 | Thanks! | Direct |
| A3 | Can I cancel an order? | RAG (policy question, no ID) |
| A4 | Can I cancel order ORD1006? (CUST003) | Tool |

### Integration (multi-turn and failures)

| # | Steps | Expected |
|---|---|---|
| I1 | "Where is my order?" then "ORD1001" | Second turn calls `get_order_status` with ORD1001 |
| I2 | "What's your return policy?" then "what about electronics?" | Second turn is RAG with a rewritten search query about electronics returns |
| I3 | Put a wrong key in `.env`, restart, send a message | Red error: "The OpenAI API key is invalid..." (no stack trace) |
| I4 | Remove the key from `.env`, restart | Red banner about the missing key, chat input disabled |
| I5 | Rename `data/knowledge_base` temporarily and delete `vector_store/`, restart | Red "Knowledge base could not be loaded" banner. Rename it back afterwards. |
| I6 | Disconnect from the internet, send a message | "I couldn't reach the OpenAI API..." |
| I7 | Press Enter on an empty message | Nothing is sent |

Tool failure handling is covered by the automated tests (`test_missing_required_argument_does_not_crash`, `test_execute_unknown_tool`).
