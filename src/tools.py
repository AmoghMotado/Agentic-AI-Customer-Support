"""Simulated customer-support tools.

These functions read and write the SQLite database (src/db.py) instead of a
real e-commerce backend. Every tool:
  1. takes a business_id and only ever touches that business's rows,
  2. validates its inputs in code (the LLM is never trusted blindly),
  3. returns a dict with "ok": True/False instead of raising, so a bad input
     becomes a helpful message rather than a crash.

The LLM sees the tool *schemas* defined at the bottom of this file
(LLM_TOOLS). The graph then calls execute_tool(), which runs the real
function with validated arguments and the business ID from the app session.
"""

import logging
from datetime import datetime

from langchain_core.tools import tool

from src import db
from src.db import CUSTOMER_ID_PATTERN, ORDER_ID_PATTERN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def normalize_order_id(order_id) -> str:
    return str(order_id or "").strip().upper()


def normalize_customer_id(customer_id) -> str:
    return str(customer_id or "").strip().upper()


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _check_order_id(order_id: str):
    """Return an error dict if the order ID is missing or malformed, else None."""
    if not order_id:
        return _error("Order ID is missing. Ask the customer for their order ID (e.g. ORD1001).")
    if not ORDER_ID_PATTERN.match(order_id):
        return _error(
            f"'{order_id}' is not a valid order ID. Order IDs look like ORD1001."
        )
    return None


def _check_customer_id(business_id: str, customer_id: str):
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
    if db.get_customer(business_id, customer_id) is None:
        return _error(f"No customer found with ID {customer_id}.")
    return None


# ---------------------------------------------------------------------------
# The actual tools
# ---------------------------------------------------------------------------
def get_order_status(business_id: str, order_id: str, customer_id: str = "") -> dict:
    """Look up an order. If a customer ID is known, the order must belong to it."""
    order_id = normalize_order_id(order_id)
    customer_id = normalize_customer_id(customer_id)

    problem = _check_order_id(order_id)
    if problem:
        return problem

    order = db.get_order(business_id, order_id)
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


def get_customer_details(business_id: str, customer_id: str) -> dict:
    """Return a customer's profile and a short list of their orders."""
    customer_id = normalize_customer_id(customer_id)
    problem = _check_customer_id(business_id, customer_id)
    if problem:
        return problem

    customer = db.get_customer(business_id, customer_id)
    customer.pop("business_id", None)
    orders = [
        {"order_id": o["order_id"], "product": o["product"], "status": o["status"]}
        for o in db.list_orders(business_id, customer_id)
    ]
    return {"ok": True, **customer, "orders": orders}


def create_support_ticket(business_id: str, customer_id: str, issue: str, order_id: str = "") -> dict:
    """Create a support ticket and save it in the database."""
    customer_id = normalize_customer_id(customer_id)
    order_id = normalize_order_id(order_id)
    issue = str(issue or "").strip()

    problem = _check_customer_id(business_id, customer_id)
    if problem:
        return problem
    if len(issue) < 5:
        return _error("The issue description is missing or too short. Ask the customer what went wrong.")

    if order_id:
        problem = _check_order_id(order_id)
        if problem:
            return problem
        order = db.get_order(business_id, order_id)
        if order is None:
            return _error(f"No order found with ID {order_id}.")
        if order["customer_id"] != customer_id:
            return _error(f"Order {order_id} is not linked to customer {customer_id}.")

    ticket_id = db.create_ticket(business_id, customer_id, issue, order_id or None)
    return {
        "ok": True,
        "ticket_id": ticket_id,
        "status": "Open",
        "message": "Support ticket created. A human agent will follow up soon.",
    }


def cancel_order(business_id: str, order_id: str, customer_id: str) -> dict:
    """Cancel an order, only if it belongs to the customer and is still Processing."""
    order_id = normalize_order_id(order_id)
    customer_id = normalize_customer_id(customer_id)

    problem = _check_order_id(order_id) or _check_customer_id(business_id, customer_id)
    if problem:
        return problem

    order = db.get_order(business_id, order_id)
    if order is None:
        return _error(f"No order found with ID {order_id}.")
    if order["customer_id"] != customer_id:
        return _error(f"Order {order_id} is not linked to customer {customer_id}.")
    if order["status"] != "Processing":
        return _error(
            f"Order {order_id} cannot be cancelled because its status is "
            f"'{order['status']}'. Only orders in 'Processing' status can be cancelled."
        )

    db.update_order(
        business_id,
        order_id,
        status="Cancelled",
        estimated_delivery="-",
        tracking_status=f"Cancelled by customer on {datetime.now():%Y-%m-%d}",
    )
    return {
        "ok": True,
        "order_id": order_id,
        "status": "Cancelled",
        "message": "Order cancelled. Prepaid amounts are refunded according to the refund policy.",
    }


# ---------------------------------------------------------------------------
# Tool schemas shown to the LLM
# ---------------------------------------------------------------------------
# These wrappers only define the name, description and arguments the LLM sees.
# The LLM never runs them directly: the graph calls execute_tool() instead.
# business_id is NOT in any schema: the LLM cannot see or choose the business.
# customer_id is optional for the LLM because the app usually knows it from the
# sidebar, and the app's value always wins (the LLM cannot act for someone else).

@tool("get_order_status")
def get_order_status_tool(order_id: str) -> dict:
    """Get the current status, tracking information and estimated delivery date
    of an order. Only call this when the customer has given an order ID like ORD1001.
    Never guess or invent an order ID."""
    raise NotImplementedError("Run through execute_tool()")


@tool("get_customer_details")
def get_customer_details_tool(customer_id: str = "") -> dict:
    """Get the customer's profile (name, email, membership) and a list of their orders.
    Leave customer_id empty to use the logged-in customer."""
    raise NotImplementedError("Run through execute_tool()")


@tool("create_support_ticket")
def create_support_ticket_tool(issue: str, order_id: str = "", customer_id: str = "") -> dict:
    """Create a support ticket for a problem that needs a human agent, such as a
    damaged item, a late delivery or a failed refund. issue is a short summary of
    the problem in the customer's words. order_id is optional. Leave customer_id
    empty to use the logged-in customer."""
    raise NotImplementedError("Run through execute_tool()")


@tool("cancel_order")
def cancel_order_tool(order_id: str, customer_id: str = "") -> dict:
    """Cancel an order. Only call this when the customer clearly asks to cancel a
    specific order ID. Leave customer_id empty to use the logged-in customer."""
    raise NotImplementedError("Run through execute_tool()")


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


def execute_tool(name: str, args: dict, business_id: str, session_customer_id: str = "") -> dict:
    """Safely run a tool requested by the LLM.

    - Unknown tools and unexpected arguments are rejected.
    - The business ID always comes from the app session, never from the LLM.
    - The customer ID from the app session overrides whatever the LLM sent.
    - Any unexpected exception becomes an error dict instead of a crash.
    """
    if name not in TOOL_FUNCTIONS:
        return _error(f"Unknown tool '{name}'.")
    if not isinstance(args, dict):
        return _error("Tool arguments were not valid.")
    if not business_id or db.get_business(business_id) is None:
        return _error("This assistant is not linked to a valid business.")

    # Anything not in ALLOWED_ARGS is dropped, including a "business_id" the LLM
    # might try to send.
    clean_args = {k: v for k, v in args.items() if k in ALLOWED_ARGS[name]}

    if "customer_id" in ALLOWED_ARGS[name]:
        session_customer_id = normalize_customer_id(session_customer_id)
        if session_customer_id:
            clean_args["customer_id"] = session_customer_id
        # create_support_ticket and get_customer_details require customer_id.
        clean_args.setdefault("customer_id", "")

    try:
        return TOOL_FUNCTIONS[name](business_id=business_id, **clean_args)
    except TypeError as exc:
        logger.exception("Bad arguments for tool %s", name)
        return _error(f"Invalid arguments for {name}: {exc}")
    except Exception:
        logger.exception("Tool %s failed", name)
        return _error(f"The {name} tool is temporarily unavailable. Please try again later.")
