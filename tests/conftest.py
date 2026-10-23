"""Every test gets a fresh temporary database and vector store, never the real ones."""

import pytest

from src import config, db, rag


@pytest.fixture(autouse=True)
def temp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", tmp_path / "vector_store")
    monkeypatch.setattr(rag, "_vector_stores", {})
    db.init_db()  # seeds NovaCart and GreenLeaf from data/seed/
    yield tmp_path
