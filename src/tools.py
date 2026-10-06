"""Simulated customer-support tools.

These functions read and write local JSON files instead of a real e-commerce
backend. Every tool:
  1. validates its inputs in code (the LLM is never trusted blindly),
  2. returns a dict with "ok": True/False instead of raising, so a bad input
     becomes a helpful message rather than a crash.

The LLM sees the tool *schemas* defined at the bottom of this file
(LLM_TOOLS). The graph then calls execute_tool(), which runs the real
function with validated arguments.
"""

import json
import logging
import re
import shutil
from datetime import datetime

from langchain_core.tools import tool

from src import config

logger = logging.getLogger(__name__)

ORDER_ID_PATTERN = re.compile(r"^ORD\d{4}$")
CUSTOMER_ID_PATTERN = re.compile(r"^CUST\d{3}$")


# ---------------------------------------------------------------------------
# Small JSON helpers
# ---------------------------------------------------------------------------
def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_order_id(order_id) -> str:
    return str(order_id or "").strip().upper()


def normalize_customer_id(customer_id) -> str:
    return str(customer_id or "").strip().upper()


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _find_order(order_id: str):
    for order in _read_json(config.ORDERS_FILE):
        if order["order_id"] == order_id:
            return order
    return None


def _find_customer(customer_id: str):
    for customer in _read_json(config.CUSTOMERS_FILE):
        if customer["customer_id"] == customer_id:
            return customer
    return None


def _check_order_id(order_id: str):
    """Return an error dict if the order ID is missing or malformed, else None."""
    if not order_id:
        return _error("Order ID is missing. Ask the customer for their order ID (e.g. ORD1001).")
    if not ORDER_ID_PATTERN.match(order_id):
        return _error(
            f"'{order_id}' is not a valid order ID. Order IDs look like ORD1001."
        )
    return None


def _check_customer_id(customer_id: str):
    """Return an error dict if the customer ID is missing/invalid/unknown, else None."""
    if not customer_id:
        return _error(
            "Customer ID is missing. Ask the customer to enter their customer ID "
            "(e.g. CUST001) in the sidebar or in the chat."
        )
    if not CUSTOMER_ID_PATTERN.match(customer_id):
        return _error(
            f"'{customer_id}' is not a valid customer ID. Customer IDs look like CUST001."
        )
    if _find_customer(customer_id) is None:
        return _error(f"No customer found with ID {customer_id}.")
    return None


# ---------------------------------------------------------------------------
# The actual tools
# ---------------------------------------------------------------------------
def get_order_status(order_id: str, customer_id: str = "") -> dict:
    """Look up an order. If a customer ID is known, the order must belong to it."""
    order_id = normalize_order_id(order_id)
    customer_id = normalize_customer_id(customer_id)

    problem = _check_order_id(order_id)
    if problem:
        return problem

    order = _find_order(order_id)
    if order is None:
        return _error(f"No order found with ID {order_id}. Please double-check the order ID.")

    if customer_id and order["customer_id"] != customer_id:
        # Do not reveal another customer's order details.
        return _error(f"Order {order_id} is not linked to customer {customer_id}.")

    return {
        "ok": True,
        "order_id": order["order_id"],
        "product": order["product"],
        "order_date": order["order_date"],
        "status": order["status"],
        "estimated_delivery": order["estimated_delivery"],
        "tracking_status": order["tracking_status"],
    }


def get_customer_details(customer_id: str) -> dict:
    """Return a customer's profile and a short list of their orders."""
    customer_id = normalize_customer_id(customer_id)
    problem = _check_customer_id(customer_id)
    if problem:
        return problem

    customer = _find_customer(customer_id)
    orders = [
        {"order_id": o["order_id"], "product": o["product"], "status": o["status"]}
        for o in _read_json(config.ORDERS_FILE)
        if o["customer_id"] == customer_id
    ]
    return {"ok": True, **customer, "orders": orders}


def create_support_ticket(customer_id: str, issue: str, order_id: str = "") -> dict:
    """Create a support ticket and save it to data/tickets.json."""
    customer_id = normalize_customer_id(customer_id)
    order_id = normalize_order_id(order_id)
    issue = str(issue or "").strip()

    problem = _check_customer_id(customer_id)
    if problem:
        return problem
    if len(issue) < 5:
        return _error("The issue description is missing or too short. Ask the customer what went wrong.")

    if order_id:
        problem = _check_order_id(order_id)
        if problem:
            return problem
        order = _find_order(order_id)
        if order is None:
            return _error(f"No order found with ID {order_id}.")
        if order["customer_id"] != customer_id:
            return _error(f"Order {order_id} is not linked to customer {customer_id}.")

    tickets = _read_json(config.TICKETS_FILE) if config.TICKETS_FILE.exists() else []
    ticket_id = f"TCK{1001 + len(tickets)}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "order_id": order_id or None,
        "issue": issue,
        "status": "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    tickets.append(ticket)
    _write_json(config.TICKETS_FILE, tickets)

    return {
        "ok": True,
        "ticket_id": ticket_id,
        "status": "Open",
        "message": "Support ticket created. Our team usually responds within 24 hours.",
    }


def cancel_order(order_id: str, customer_id: str) -> dict:
    """Cancel an order, only if it belongs to the customer and is still Processing."""
    order_id = normalize_order_id(order_id)
    customer_id = normalize_customer_id(customer_id)

    problem = _check_order_id(order_id) or _check_customer_id(customer_id)
    if problem:
        return problem

    orders = _read_json(config.ORDERS_FILE)
    order = next((o for o in orders if o["order_id"] == order_id), None)
    if order is None:
        return _error(f"No order found with ID {order_id}.")
    if order["customer_id"] != customer_id:
        return _error(f"Order {order_id} is not linked to customer {customer_id}.")
    if order["status"] != "Processing":
        return _error(
            f"Order {order_id} cannot be cancelled because its status is "
            f"'{order['status']}'. Only orders in 'Processing' status can be cancelled."
        )

    order["status"] = "Cancelled"
    order["estimated_delivery"] = "-"
    order["tracking_status"] = f"Cancelled by customer on {datetime.now():%Y-%m-%d}"
    _write_json(config.ORDERS_FILE, orders)
    return {
        "ok": True,
        "order_id": order_id,
        "status": "Cancelled",
        "message": "Order cancelled. Prepaid amounts are refunded to the original payment method.",
    }


def reset_demo_data():
    """Restore orders.json from data/seed and clear all tickets."""
    shutil.copyfile(config.SEED_ORDERS_FILE, config.ORDERS_FILE)
    _write_json(config.TICKETS_FILE, [])


# ---------------------------------------------------------------------------
# Tool schemas shown to the LLM
# ---------------------------------------------------------------------------
# These wrappers only define the name, description and arguments the LLM sees.
# The LLM never runs them directly: the graph calls execute_tool() instead.
# customer_id is optional for the LLM because the app usually knows it from the
# sidebar, and the app's value always wins (the LLM cannot act for someone else).

@tool("get_order_status")
def get_order_status_tool(order_id: str) -> dict:
    """Get the current status, tracking information and estimated delivery date
    of an order. Only call this when the customer has given an order ID like ORD1001.
    Never guess or invent an order ID."""
    return get_order_status(order_id)


@tool("get_customer_details")
def get_customer_details_tool(customer_id: str = "") -> dict:
    """Get the customer's profile (name, email, membership) and a list of their orders.
    Leave customer_id empty to use the logged-in customer."""
    return get_customer_details(customer_id)


@tool("create_support_ticket")
def create_support_ticket_tool(issue: str, order_id: str = "", customer_id: str = "") -> dict:
    """Create a support ticket for a problem that needs a human agent, such as a
    damaged item, a late delivery or a failed refund. issue is a short summary of
    the problem in the customer's words. order_id is optional. Leave customer_id
    empty to use the logged-in customer."""
    return create_support_ticket(customer_id, issue, order_id)


@tool("cancel_order")
def cancel_order_tool(order_id: str, customer_id: str = "") -> dict:
    """Cancel an order. Only call this when the customer clearly asks to cancel a
    specific order ID. Leave customer_id empty to use the logged-in customer."""
    return cancel_order(order_id, customer_id)


LLM_TOOLS = [
    get_order_status_tool,
    get_customer_details_tool,
    create_support_ticket_tool,
    cancel_order_tool,
]

TOOL_FUNCTIONS = {
    "get_order_status": get_order_status,
    "get_customer_details": get_customer_details,
    "create_support_ticket": create_support_ticket,
    "cancel_order": cancel_order,
}

# Arguments each tool accepts (anything else the LLM sends is dropped).
ALLOWED_ARGS = {
    "get_order_status": {"order_id", "customer_id"},
    "get_customer_details": {"customer_id"},
    "create_support_ticket": {"customer_id", "issue", "order_id"},
    "cancel_order": {"order_id", "customer_id"},
}


def execute_tool(name: str, args: dict, session_customer_id: str = "") -> dict:
    """Safely run a tool requested by the LLM.

    - Unknown tools and unexpected arguments are rejected.
    - The customer ID from the app session overrides whatever the LLM sent.
    - Any unexpected exception becomes an error dict instead of a crash.
    """
    if name not in TOOL_FUNCTIONS:
        return _error(f"Unknown tool '{name}'.")
    if not isinstance(args, dict):
        return _error("Tool arguments were not valid.")

    clean_args = {k: v for k, v in args.items() if k in ALLOWED_ARGS[name]}

    if "customer_id" in ALLOWED_ARGS[name]:
        session_customer_id = normalize_customer_id(session_customer_id)
        if session_customer_id:
            clean_args["customer_id"] = session_customer_id
        # create_support_ticket and get_customer_details require customer_id.
        clean_args.setdefault("customer_id", "")

    try:
        return TOOL_FUNCTIONS[name](**clean_args)
    except TypeError as exc:
        logger.exception("Bad arguments for tool %s", name)
        return _error(f"Invalid arguments for {name}: {exc}")
    except Exception:
        logger.exception("Tool %s failed", name)
        return _error(f"The {name} tool is temporarily unavailable. Please try again later.")
