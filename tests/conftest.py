"""Make every test use temporary copies of the JSON data, never the real files."""

import shutil

import pytest

from src import config


@pytest.fixture(autouse=True)
def temp_data(tmp_path, monkeypatch):
    for name in ("orders.json", "customers.json"):
        shutil.copyfile(config.DATA_DIR / name, tmp_path / name)
    shutil.copyfile(config.SEED_ORDERS_FILE, tmp_path / "orders.json")
    (tmp_path / "tickets.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(config, "ORDERS_FILE", tmp_path / "orders.json")
    monkeypatch.setattr(config, "CUSTOMERS_FILE", tmp_path / "customers.json")
    monkeypatch.setattr(config, "TICKETS_FILE", tmp_path / "tickets.json")
    yield tmp_path
