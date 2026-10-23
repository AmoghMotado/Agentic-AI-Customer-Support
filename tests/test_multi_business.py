"""Tests that each business only ever sees its own data.

NovaCart and GreenLeaf both have a customer called CUST001 (different people),
which makes cross-business leaks easy to spot. No API key or model download needed.
"""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src import db, rag
from src.tools import cancel_order, execute_tool, get_customer_details, get_order_status


def test_demo_businesses_are_seeded():
    ids = {b["business_id"] for b in db.list_businesses()}
    assert {"novacart", "greenleaf"} <= ids


def test_same_customer_id_is_a_different_person_per_business():
    assert get_customer_details("novacart", "CUST001")["name"] == "Aarav Sharma"
    assert get_customer_details("greenleaf", "CUST001")["name"] == "Meera Iyer"


def test_order_of_another_business_is_not_found():
    # ORD1001 is a NovaCart order. GreenLeaf must not see it.
    result = get_order_status("greenleaf", "ORD1001")
    assert result["ok"] is False
    assert "No order found" in result["error"]


def test_cannot_cancel_another_business_order():
    # ORD1003 is a cancellable NovaCart order of CUST002. GreenLeaf also has a CUST002.
    assert cancel_order("greenleaf", "ORD1003", "CUST002")["ok"] is False
    assert get_order_status("novacart", "ORD1003")["status"] == "Processing"


def test_tickets_are_numbered_and_stored_per_business():
    a = execute_tool("create_support_ticket", {"issue": "Item arrived damaged"}, "novacart", "CUST001")
    b = execute_tool("create_support_ticket", {"issue": "Spoiled milk in order"}, "greenleaf", "CUST001")
    assert a["ticket_id"] == b["ticket_id"] == "TCK1001"
    assert [t["issue"] for t in db.list_tickets("novacart")] == ["Item arrived damaged"]
    assert [t["issue"] for t in db.list_tickets("greenleaf")] == ["Spoiled milk in order"]


def test_llm_cannot_choose_the_business():
    # The LLM sends business_id="novacart", but the session is GreenLeaf. It is ignored.
    result = execute_tool(
        "get_order_status", {"order_id": "ORD1001", "business_id": "novacart"}, "greenleaf"
    )
    assert result["ok"] is False


def test_unknown_business_is_rejected():
    assert execute_tool("get_order_status", {"order_id": "ORD1001"}, "no-such-shop")["ok"] is False


def test_reset_demo_data_only_touches_one_business():
    cancel_order("novacart", "ORD1003", "CUST002")
    cancel_order("greenleaf", "ORD5003", "CUST002")
    db.reset_demo_data("novacart")
    assert get_order_status("novacart", "ORD1003")["status"] == "Processing"
    assert get_order_status("greenleaf", "ORD5003")["status"] == "Cancelled"


def test_deleting_a_business_deletes_only_its_data():
    db.delete_business("greenleaf")
    assert db.get_business("greenleaf") is None
    assert db.list_orders("greenleaf") == [] and db.list_documents("greenleaf") == []
    assert len(db.list_orders("novacart")) == 6


# ---------- Admin input validation ----------
@pytest.mark.parametrize("bad_id", ["", "A", "Has Spaces", "../etc", "x" * 31])
def test_invalid_business_ids_are_rejected(bad_id):
    with pytest.raises(db.DataError):
        db.create_business(bad_id, "Some Shop")


def test_duplicate_business_is_rejected():
    with pytest.raises(db.DataError):
        db.create_business("novacart", "Another NovaCart")


def test_new_business_with_uploaded_data():
    db.create_business("city-bakery", "City Bakery", description="a bakery in Pune")
    db.add_document("city-bakery", "C:\\Users\\me\\faq.txt", "Cakes need 24 hours notice.")
    assert [d["filename"] for d in db.list_documents("city-bakery")] == ["faq.txt"]
    db.replace_customers("city-bakery", [{"customer_id": "cust001", "name": "Asha"}])
    db.replace_orders("city-bakery", [
        {"order_id": "ORD1001", "customer_id": "CUST001", "product": "Cake", "status": "Processing"}
    ])
    assert get_order_status("city-bakery", "ORD1001", "CUST001")["product"] == "Cake"
    # NovaCart's ORD1001 is untouched
    assert get_order_status("novacart", "ORD1001")["product"] == "Wireless Bluetooth Headphones"


def test_order_for_unknown_customer_is_rejected():
    with pytest.raises(db.DataError):
        db.replace_orders("greenleaf", [
            {"order_id": "ORD5009", "customer_id": "CUST999", "product": "Milk", "status": "Processing"}
        ])


def test_api_key_is_never_returned_with_business_settings():
    db.update_business("greenleaf", openai_api_key="sk-greenleaf-secret")
    business = db.get_business("greenleaf")
    assert "openai_api_key" not in business and business["has_own_api_key"] is True
    assert db.get_business_api_key("greenleaf") == "sk-greenleaf-secret"
    assert db.get_business_api_key("novacart") == ""


# ---------- Retrieval isolation (fake embeddings, no download) ----------
def test_each_business_searches_only_its_own_documents(monkeypatch):
    monkeypatch.setattr(rag, "get_embeddings", lambda: DeterministicFakeEmbedding(size=64))
    novacart_files = {d["filename"] for d in db.list_documents("novacart")}
    greenleaf_files = {d["filename"] for d in db.list_documents("greenleaf")}

    novacart_hits = rag.retrieve("novacart", "What is your refund policy?", k=10)
    greenleaf_hits = rag.retrieve("greenleaf", "What is your refund policy?", k=10)

    assert {h["source"] for h in novacart_hits} <= novacart_files
    assert {h["source"] for h in greenleaf_hits} <= greenleaf_files
    assert rag.index_dir("novacart") != rag.index_dir("greenleaf")
    assert (rag.index_dir("greenleaf") / "index.faiss").exists()
