"""RAG tests for loading and splitting (fast, no model download).

The retrieval-quality tests download the embedding model on first run, so they
are skipped unless you run:  python -m pytest -v -m slow
"""

import pytest

from src import config, db, rag


def test_documents_load_from_database():
    docs = rag.load_documents("novacart")
    sources = {d.metadata["source"] for d in docs}
    assert "refund_policy.txt" in sources
    assert len(docs) == len(list((config.SEED_DIR / "novacart" / "knowledge_base").glob("*.txt")))


def test_chunks_are_small_and_numbered():
    docs = rag.load_documents("novacart")
    chunks = rag.split_documents(docs)
    assert len(chunks) > len(docs)
    assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)
    assert all(c.metadata["chunk"] >= 1 for c in chunks)


def test_business_without_documents():
    db.create_business("empty-shop", "Empty Shop")
    with pytest.raises(rag.KnowledgeBaseError):
        rag.load_documents("empty-shop")


@pytest.mark.slow
@pytest.mark.parametrize(
    "question, expected_source",
    [
        ("What is your refund policy?", "refund_policy.txt"),
        ("How many days do I have to return an item?", "return_policy.txt"),
        ("How long does delivery take?", "shipping_policy.txt"),
        ("Which payment methods do you accept?", "payment_policy.txt"),
        ("My product arrived broken, what do I do?", "damaged_product_policy.txt"),
    ],
)
def test_retrieval_finds_right_document(question, expected_source):
    hits = rag.retrieve("novacart", question, k=3)
    assert expected_source in [h["source"] for h in hits]
