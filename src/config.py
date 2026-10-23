"""Central configuration: paths, model names and settings.

Everything that might change between machines lives here, so the rest of the
code never hardcodes a path or an API key.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains app.py
BASE_DIR = Path(__file__).resolve().parent.parent

# Read variables from .env into the environment (does nothing if .env is missing)
load_dotenv(BASE_DIR / ".env")

# ---------- Paths ----------
DATA_DIR = BASE_DIR / "data"
# SQLite database with every business, its documents, customers, orders and tickets.
# Created automatically on first run (and seeded from data/seed/).
DB_PATH = DATA_DIR / "app.db"
# One folder per demo business: business.json, customers.json, orders.json, knowledge_base/
SEED_DIR = DATA_DIR / "seed"
# One FAISS index per business: vector_store/<business_id>/
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
# Business shown first in the chat app
DEFAULT_BUSINESS_ID = os.getenv("DEFAULT_BUSINESS_ID", "novacart").strip().lower()

# ---------- Admin page ----------
# If set, the Admin page asks for this password. Leave empty for local demos.
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

# ---------- OpenAI ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()

# ---------- Embeddings ----------
# "huggingface" = free local model (default), "openai" = OpenAI embeddings API
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
).strip()
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
).strip()

# ---------- RAG settings ----------
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 80  # characters shared between neighbouring chunks
TOP_K = 4  # number of chunks retrieved per question

# ---------- Agent settings ----------
MAX_TOOL_ROUNDS = 3  # max tool-call rounds per user message (stops runaway loops)
MAX_HISTORY_MESSAGES = 10  # how many past chat messages are sent to the LLM


def key_looks_real(key: str) -> bool:
    """True if a key is filled in and is not the placeholder from .env.example."""
    key = (key or "").strip()
    return bool(key) and not key.startswith("sk-your")


def api_key_is_set() -> bool:
    """True if a real-looking OpenAI key is configured in .env."""
    return key_looks_real(OPENAI_API_KEY)
