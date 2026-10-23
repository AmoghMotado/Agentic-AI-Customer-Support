"""SQLite storage for every business (tenant) the assistant serves.

One small database file (data/app.db) holds all businesses. Every table except
`businesses` has a `business_id` column, and every query in this file filters
on it. That one rule is what keeps one business's data away from another's.

Tables:
    businesses  settings and branding for each business
    documents   knowledge-base text for RAG (one row per uploaded file)
    customers   each business's customers
    orders      each business's orders
    tickets     support tickets created by the assistant

The database is created and seeded automatically from data/seed/ the first
time init_db() runs. Delete data/app.db to start again from the seed data.
"""

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PureWindowsPath

from src import config

# business IDs are used in URLs and folder names, so keep them simple.
BUSINESS_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,29}$")
ORDER_ID_PATTERN = re.compile(r"^ORD\d{4}$")
CUSTOMER_ID_PATTERN = re.compile(r"^CUST\d{3}$")

# Fields an admin may change on a business (business_id itself never changes).
BUSINESS_FIELDS = ("name", "description", "logo", "accent_color", "currency", "openai_api_key")

SCHEMA = """
CREATE TABLE IF NOT EXISTS businesses (
    business_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    logo            TEXT NOT NULL DEFAULT '💬',
    accent_color    TEXT NOT NULL DEFAULT '#4f46e5',
    currency        TEXT NOT NULL DEFAULT 'INR',
    openai_api_key  TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    business_id  TEXT NOT NULL REFERENCES businesses(business_id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    content      TEXT NOT NULL,
    uploaded_at  TEXT NOT NULL,
    PRIMARY KEY (business_id, filename)
);

CREATE TABLE IF NOT EXISTS customers (
    business_id   TEXT NOT NULL REFERENCES businesses(business_id) ON DELETE CASCADE,
    customer_id   TEXT NOT NULL,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL DEFAULT '',
    phone         TEXT NOT NULL DEFAULT '',
    city          TEXT NOT NULL DEFAULT '',
    membership    TEXT NOT NULL DEFAULT '',
    member_since  TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (business_id, customer_id)
);

CREATE TABLE IF NOT EXISTS orders (
    business_id         TEXT NOT NULL REFERENCES businesses(business_id) ON DELETE CASCADE,
    order_id            TEXT NOT NULL,
    customer_id         TEXT NOT NULL,
    product             TEXT NOT NULL,
    amount              REAL NOT NULL DEFAULT 0,
    order_date          TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL,
    estimated_delivery  TEXT NOT NULL DEFAULT '-',
    tracking_status     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (business_id, order_id)
);

CREATE TABLE IF NOT EXISTS tickets (
    business_id  TEXT NOT NULL REFERENCES businesses(business_id) ON DELETE CASCADE,
    ticket_id    TEXT NOT NULL,
    customer_id  TEXT NOT NULL,
    order_id     TEXT,
    issue        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'Open',
    created_at   TEXT NOT NULL,
    PRIMARY KEY (business_id, ticket_id)
);
"""


class DataError(ValueError):
    """Raised when admin input (a business, a customers file, ...) is invalid."""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


@contextmanager
def connect():
    """Open the database, commit if the block succeeds, and always close it.

    A new short-lived connection per call is simple and safe with Streamlit,
    which runs each browser session in its own thread.
    """
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["name"]
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Setup and seeding
# ---------------------------------------------------------------------------
def init_db():
    """Create the tables, and load the demo businesses if the database is empty."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM businesses").fetchone()[0]
    if count == 0:
        # The default business (NovaCart) first, then the others alphabetically.
        folders = sorted(config.SEED_DIR.iterdir(),
                         key=lambda f: (f.name != config.DEFAULT_BUSINESS_ID, f.name))
        for folder in folders:
            if (folder / "business.json").exists():
                seed_business(folder)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seed_business(folder: Path):
    """Load one demo business from a seed folder (business.json, customers, orders, docs)."""
    info = _read_json(folder / "business.json")
    business_id = info.pop("business_id")
    create_business(business_id, **info)
    if (folder / "customers.json").exists():
        replace_customers(business_id, _read_json(folder / "customers.json"))
    if (folder / "orders.json").exists():
        replace_orders(business_id, _read_json(folder / "orders.json"))
    for path in sorted((folder / "knowledge_base").glob("*.txt")):
        # encoding="utf-8" matters on Windows, where the default encoding differs.
        add_document(business_id, path.name, path.read_text(encoding="utf-8"))


def reset_demo_data(business_id: str):
    """Delete this business's tickets and restore its orders from the seed folder."""
    with connect() as conn:
        conn.execute("DELETE FROM tickets WHERE business_id = ?", (business_id,))
    seed_orders = config.SEED_DIR / business_id / "orders.json"
    if seed_orders.exists():
        replace_orders(business_id, _read_json(seed_orders))


# ---------------------------------------------------------------------------
# Businesses
# ---------------------------------------------------------------------------
def _public_business(row) -> dict:
    """A business as a dict WITHOUT its API key (safe to show in the UI)."""
    business = dict(row)
    business["has_own_api_key"] = bool(business.pop("openai_api_key", ""))
    return business


def create_business(business_id: str, name: str, **fields) -> dict:
    business_id = str(business_id or "").strip().lower()
    name = str(name or "").strip()
    if not BUSINESS_ID_PATTERN.match(business_id):
        raise DataError(
            "Business ID must be 2 to 30 characters: lowercase letters, numbers and dashes "
            "(for example 'greenleaf' or 'city-bakery')."
        )
    if not name:
        raise DataError("Business name is required.")
    if get_business(business_id):
        raise DataError(f"A business with ID '{business_id}' already exists.")

    values = {k: str(v).strip() for k, v in fields.items() if k in BUSINESS_FIELDS and v is not None}
    columns = ["business_id", "name", "created_at", *values]
    with connect() as conn:
        conn.execute(
            f"INSERT INTO businesses ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            [business_id, name, _now(), *values.values()],
        )
    return get_business(business_id)


def update_business(business_id: str, **fields) -> dict:
    """Change settings. Pass openai_api_key="" to remove a business's own key."""
    values = {k: str(v).strip() for k, v in fields.items() if k in BUSINESS_FIELDS and v is not None}
    if "name" in values and not values["name"]:
        raise DataError("Business name is required.")
    if values:
        assignments = ", ".join(f"{k} = ?" for k in values)
        with connect() as conn:
            conn.execute(
                f"UPDATE businesses SET {assignments} WHERE business_id = ?",
                [*values.values(), business_id],
            )
    return get_business(business_id)


def get_business(business_id: str):
    """The business's settings (without its API key), or None if it doesn't exist."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM businesses WHERE business_id = ?", (business_id,)
        ).fetchone()
    return _public_business(row) if row else None


def get_business_api_key(business_id: str) -> str:
    """The business's own OpenAI key, or "" if it uses the key from .env."""
    with connect() as conn:
        row = conn.execute(
            "SELECT openai_api_key FROM businesses WHERE business_id = ?", (business_id,)
        ).fetchone()
    return row["openai_api_key"] if row else ""


def list_businesses() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM businesses ORDER BY rowid").fetchall()
    return [_public_business(r) for r in rows]


def delete_business(business_id: str):
    """Delete a business and (through ON DELETE CASCADE) all of its data."""
    with connect() as conn:
        conn.execute("DELETE FROM businesses WHERE business_id = ?", (business_id,))


# ---------------------------------------------------------------------------
# Knowledge-base documents
# ---------------------------------------------------------------------------
def add_document(business_id: str, filename: str, content: str):
    """Add a document, or replace the one with the same file name."""
    # Keep only the file name: PureWindowsPath drops folders written with / or \
    filename = PureWindowsPath(str(filename or "")).name.strip()
    content = str(content or "").strip()
    if not filename:
        raise DataError("Document needs a file name.")
    if not content:
        raise DataError(f"'{filename}' is empty.")
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (business_id, filename, content, uploaded_at) "
            "VALUES (?, ?, ?, ?)",
            (business_id, filename, content, _now()),
        )


def list_documents(business_id: str) -> list[dict]:
    """Documents with their full text, sorted by file name."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT filename, content, uploaded_at FROM documents "
            "WHERE business_id = ? ORDER BY filename",
            (business_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(business_id: str, filename: str):
    with connect() as conn:
        conn.execute(
            "DELETE FROM documents WHERE business_id = ? AND filename = ?",
            (business_id, filename),
        )


# ---------------------------------------------------------------------------
# Customers and orders
# ---------------------------------------------------------------------------
def replace_customers(business_id: str, customers: list):
    """Replace ALL of a business's customers with a new list."""
    if not isinstance(customers, list):
        raise DataError("Customers file must be a JSON list.")
    rows, seen = [], set()
    for c in customers:
        if not isinstance(c, dict):
            raise DataError("Each customer must be a JSON object.")
        customer_id = str(c.get("customer_id", "")).strip().upper()
        if not CUSTOMER_ID_PATTERN.match(customer_id):
            raise DataError(f"Invalid customer_id '{customer_id}'. Use the format CUST001.")
        if customer_id in seen:
            raise DataError(f"Duplicate customer_id {customer_id}.")
        if not str(c.get("name", "")).strip():
            raise DataError(f"Customer {customer_id} has no name.")
        seen.add(customer_id)
        rows.append((
            business_id, customer_id, str(c["name"]).strip(),
            *(str(c.get(k, "")) for k in ("email", "phone", "city", "membership", "member_since")),
        ))

    with connect() as conn:
        conn.execute("DELETE FROM customers WHERE business_id = ?", (business_id,))
        conn.executemany(
            "INSERT INTO customers (business_id, customer_id, name, email, phone, city, "
            "membership, member_since) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        # Orders of customers who are no longer in the list are removed too.
        conn.execute(
            "DELETE FROM orders WHERE business_id = ? AND customer_id NOT IN "
            "(SELECT customer_id FROM customers WHERE business_id = ?)",
            (business_id, business_id),
        )
    return len(rows)


def replace_orders(business_id: str, orders: list):
    """Replace ALL of a business's orders. Every order must belong to one of its customers."""
    if not isinstance(orders, list):
        raise DataError("Orders file must be a JSON list.")
    known_customers = {c["customer_id"] for c in list_customers(business_id)}
    rows, seen = [], set()
    for o in orders:
        if not isinstance(o, dict):
            raise DataError("Each order must be a JSON object.")
        order_id = str(o.get("order_id", "")).strip().upper()
        customer_id = str(o.get("customer_id", "")).strip().upper()
        if not ORDER_ID_PATTERN.match(order_id):
            raise DataError(f"Invalid order_id '{order_id}'. Use the format ORD1001.")
        if order_id in seen:
            raise DataError(f"Duplicate order_id {order_id}.")
        if customer_id not in known_customers:
            raise DataError(f"Order {order_id} belongs to unknown customer '{customer_id}'. "
                            "Upload the customers first.")
        if not str(o.get("product", "")).strip() or not str(o.get("status", "")).strip():
            raise DataError(f"Order {order_id} needs a product and a status.")
        try:
            amount = float(o.get("amount", 0) or 0)
        except (TypeError, ValueError):
            raise DataError(f"Order {order_id} has an invalid amount.") from None
        seen.add(order_id)
        rows.append((
            business_id, order_id, customer_id, str(o["product"]).strip(), amount,
            str(o.get("order_date", "")), str(o["status"]).strip(),
            str(o.get("estimated_delivery", "-")), str(o.get("tracking_status", "")),
        ))

    with connect() as conn:
        conn.execute("DELETE FROM orders WHERE business_id = ?", (business_id,))
        conn.executemany(
            "INSERT INTO orders (business_id, order_id, customer_id, product, amount, order_date, "
            "status, estimated_delivery, tracking_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def list_customers(business_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM customers WHERE business_id = ? ORDER BY customer_id", (business_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_customer(business_id: str, customer_id: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE business_id = ? AND customer_id = ?",
            (business_id, customer_id),
        ).fetchone()
    return dict(row) if row else None


def list_orders(business_id: str, customer_id: str = "") -> list[dict]:
    """All of a business's orders, or only one customer's orders."""
    query = "SELECT * FROM orders WHERE business_id = ?"
    params = [business_id]
    if customer_id:
        query += " AND customer_id = ?"
        params.append(customer_id)
    with connect() as conn:
        rows = conn.execute(query + " ORDER BY order_id", params).fetchall()
    return [dict(r) for r in rows]


def get_order(business_id: str, order_id: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE business_id = ? AND order_id = ?",
            (business_id, order_id),
        ).fetchone()
    return dict(row) if row else None


def update_order(business_id: str, order_id: str, **fields):
    allowed = {"status", "estimated_delivery", "tracking_status"}
    values = {k: v for k, v in fields.items() if k in allowed}
    if not values:
        return
    assignments = ", ".join(f"{k} = ?" for k in values)
    with connect() as conn:
        conn.execute(
            f"UPDATE orders SET {assignments} WHERE business_id = ? AND order_id = ?",
            [*values.values(), business_id, order_id],
        )


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------
def create_ticket(business_id: str, customer_id: str, issue: str, order_id=None) -> str:
    """Save a ticket and return its ID. Ticket numbers start at TCK1001 for every business."""
    with connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE business_id = ?", (business_id,)
        ).fetchone()[0]
        ticket_id = f"TCK{1001 + count}"
        conn.execute(
            "INSERT INTO tickets (business_id, ticket_id, customer_id, order_id, issue, status, "
            "created_at) VALUES (?, ?, ?, ?, ?, 'Open', ?)",
            (business_id, ticket_id, customer_id, order_id, issue, _now()),
        )
    return ticket_id


def list_tickets(business_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tickets WHERE business_id = ? ORDER BY created_at, ticket_id",
            (business_id,),
        ).fetchall()
    return [dict(r) for r in rows]
