"""Tool tests. These run without an OpenAI key and cost nothing.

Run:  python -m pytest -v
"""

import json

from src import config
from src.tools import (
    cancel_order,
    create_support_ticket,
    execute_tool,
    get_customer_details,
    get_order_status,
)


# ---------- get_order_status ----------
def test_valid_order():
    result = get_order_status("ORD1001")
    assert result["ok"] is True
    assert result["status"] == "Shipped"


def test_order_id_is_normalised():
    assert get_order_status("  ord1002 ")["status"] == "Delivered"


def test_unknown_order():
    result = get_order_status("ORD9999")
    assert result["ok"] is False
    assert "No order found" in result["error"]


def test_invalid_order_format():
    result = get_order_status("12345")
    assert result["ok"] is False
    assert "not a valid order ID" in result["error"]


def test_missing_order_id():
    result = get_order_status("")
    assert result["ok"] is False
    assert "missing" in result["error"]


def test_order_of_another_customer_is_hidden():
    result = get_order_status("ORD1003", customer_id="CUST001")
    assert result["ok"] is False
    assert "not linked" in result["error"]


# ---------- get_customer_details ----------
def test_customer_details():
    result = get_customer_details("CUST002")
    assert result["ok"] is True
    assert result["name"] == "Priya Nair"
    assert {o["order_id"] for o in result["orders"]} == {"ORD1003", "ORD1004"}


def test_missing_customer_id():
    result = get_customer_details("")
    assert result["ok"] is False
    assert "missing" in result["error"]


def test_unknown_customer():
    assert get_customer_details("CUST999")["ok"] is False


# ---------- create_support_ticket ----------
def test_ticket_creation():
    result = create_support_ticket("CUST001", "Package arrived damaged", "ORD1002")
    assert result["ok"] is True
    assert result["ticket_id"] == "TCK1001"
    saved = json.loads(config.TICKETS_FILE.read_text(encoding="utf-8"))
    assert saved[0]["order_id"] == "ORD1002"

    second = create_support_ticket("CUST001", "Another issue here")
    assert second["ticket_id"] == "TCK1002"


def test_ticket_missing_customer():
    result = create_support_ticket("", "Package arrived damaged")
    assert result["ok"] is False


def test_ticket_missing_issue():
    assert create_support_ticket("CUST001", "")["ok"] is False


def test_ticket_for_someone_elses_order():
    assert create_support_ticket("CUST001", "Damaged item", "ORD1003")["ok"] is False


# ---------- cancel_order ----------
def test_cancel_processing_order():
    result = cancel_order("ORD1003", "CUST002")
    assert result["ok"] is True
    assert get_order_status("ORD1003")["status"] == "Cancelled"


def test_cannot_cancel_shipped_order():
    result = cancel_order("ORD1001", "CUST001")
    assert result["ok"] is False
    assert "cannot be cancelled" in result["error"]


def test_cannot_cancel_other_customers_order():
    assert cancel_order("ORD1003", "CUST001")["ok"] is False


# ---------- execute_tool (what the graph calls) ----------
def test_execute_unknown_tool():
    assert execute_tool("delete_database", {})["ok"] is False


def test_session_customer_overrides_llm():
    # The LLM tries to act as CUST002, but the signed-in customer is CUST001.
    result = execute_tool("cancel_order", {"order_id": "ORD1003", "customer_id": "CUST002"}, "CUST001")
    assert result["ok"] is False
    assert "not linked" in result["error"]


def test_extra_arguments_are_ignored():
    result = execute_tool("get_order_status", {"order_id": "ORD1001", "evil": "x"})
    assert result["ok"] is True


def test_missing_required_argument_does_not_crash():
    result = execute_tool("get_order_status", {})
    assert result["ok"] is False
