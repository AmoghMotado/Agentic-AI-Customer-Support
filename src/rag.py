"""RAG pipeline: load documents -> split -> embed -> FAISS -> retrieve.

Every business has its OWN FAISS index in vector_store/<business_id>/, built
only from that business's documents. A search for one business can never
return another business's text, because it searches a different index.

Run this file directly to (re)build the indexes of all businesses:
    python -m src.rag
"""

import json
import logging
import shutil

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config, db

logger = logging.getLogger(__name__)

# Loaded indexes, one per business: {"novacart": FAISS, ...}.
# Loading from disk is slow, so each index is loaded once and reused.
_vector_stores = {}
# The embedding model is shared by all businesses and also loaded only once.
_embeddings = None


class KnowledgeBaseError(Exception):
    """Raised when the knowledge base or vector store cannot be used."""


def get_embeddings():
    """Return the embedding model chosen in .env (EMBEDDING_PROVIDER), loaded once."""
    global _embeddings
    if _embeddings is None:
        _embeddings = _create_embeddings()
    return _embeddings


def _create_embeddings():
    if config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not config.api_key_is_set():
            raise KnowledgeBaseError(
                "EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is not set."
            )
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY
        )

    # Default: free local model. Downloaded once (~90 MB) and cached on disk.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.HF_EMBEDDING_MODEL,
        # Normalised vectors make FAISS distances comparable (0 = identical).
        encode_kwargs={"normalize_embeddings": True},
    )


def _embedding_model_name() -> str:
    if config.EMBEDDING_PROVIDER == "openai":
        return f"openai:{config.OPENAI_EMBEDDING_MODEL}"
    return f"huggingface:{config.HF_EMBEDDING_MODEL}"


def index_dir(business_id: str):
    """Folder that holds one business's FAISS index."""
    return config.VECTOR_STORE_DIR / business_id


def load_documents(business_id: str):
    """Load one business's knowledge-base documents from the database."""
    rows = db.list_documents(business_id)
    if not rows:
        raise KnowledgeBaseError(
            "This business has no knowledge-base documents yet. "
            "Upload some on the Admin page."
        )
    return [
        Document(page_content=row["content"], metadata={"source": row["filename"]})
        for row in rows
    ]


def split_documents(documents):
    """Split documents into overlapping chunks small enough to embed well."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # Try to split on blank lines first, then lines, then sentences.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Number chunks per source file so debug mode can show "refund_policy.txt #2".
    counters = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk"] = counters[source]
    return chunks


def build_vector_store(business_id: str):
    """Build one business's FAISS index from scratch and save it to disk."""
    documents = load_documents(business_id)
    chunks = split_documents(documents)
    logger.info("[%s] Embedding %d chunks from %d documents...",
                business_id, len(chunks), len(documents))

    folder = index_dir(business_id)
    try:
        store = FAISS.from_documents(chunks, get_embeddings())
        folder.mkdir(parents=True, exist_ok=True)
        store.save_local(str(folder))
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(f"Could not build the vector store: {exc}") from exc

    (folder / "index_info.json").write_text(
        json.dumps(
            {
                "business_id": business_id,
                "embedding_model": _embedding_model_name(),
                "documents": len(documents),
                "chunks": len(chunks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _vector_stores[business_id] = store
    return store


def index_info(business_id: str):
    """What the saved index was built from, or None if there is no index."""
    try:
        return json.loads((index_dir(business_id) / "index_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _index_is_usable(business_id: str) -> bool:
    """The saved index exists and was built with the current embedding model."""
    if not (index_dir(business_id) / "index.faiss").exists():
        return False
    info = index_info(business_id) or {}
    return info.get("embedding_model") == _embedding_model_name()


def get_vector_store(business_id: str):
    """Load a business's FAISS index from disk, building it first if needed."""
    if business_id in _vector_stores:
        return _vector_stores[business_id]

    if not _index_is_usable(business_id):
        logger.info("[%s] No usable vector store found, building a new one.", business_id)
        return build_vector_store(business_id)

    try:
        # allow_dangerous_deserialization: FAISS metadata is stored with pickle.
        # Safe here because we only ever load an index this app created itself.
        _vector_stores[business_id] = FAISS.load_local(
            str(index_dir(business_id)),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        logger.warning("[%s] Saved index could not be loaded (%s). Rebuilding.", business_id, exc)
        return build_vector_store(business_id)
    return _vector_stores[business_id]


def delete_vector_store(business_id: str):
    """Forget and delete a business's index (it is rebuilt on the next question)."""
    _vector_stores.pop(business_id, None)
    folder = index_dir(business_id)
    if folder.exists():
        shutil.rmtree(folder)


def retrieve(business_id: str, query: str, k: int = config.TOP_K):
    """Return the k most relevant chunks for a query, from ONE business's index.

    Each result is a dict: {"source", "chunk", "score", "content"}.
    score is a distance: lower means more similar.
    """
    if not query or not query.strip():
        return []
    try:
        store = get_vector_store(business_id)
        results = store.similarity_search_with_score(query, k=k)
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(f"Vector search failed: {exc}") from exc

    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "chunk": doc.metadata.get("chunk", 0),
            "score": round(float(score), 3),
            "content": doc.page_content,
        }
        for doc, score in results
    ]


if __name__ == "__main__":
    # Manual check: build every business's index and run one test query on each.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    db.init_db()
    question = "What is your refund policy?"
    for business in db.list_businesses():
        business_id = business["business_id"]
        try:
            build_vector_store(business_id)
        except KnowledgeBaseError as exc:
            print(f"\n{business['name']}: skipped ({exc})")
            continue
        info = index_info(business_id)
        print(f"\n{business['name']} ({business_id}): "
              f"{info['documents']} documents, {info['chunks']} chunks, {info['embedding_model']}")
        print(f"Test query: {question}")
        for hit in retrieve(business_id, question, k=3):
            preview = hit["content"].replace("\n", " ")[:80]
            print(f"  {hit['source']} #{hit['chunk']} (distance {hit['score']}): {preview}...")
