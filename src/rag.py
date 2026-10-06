"""RAG pipeline: load documents -> split -> embed -> FAISS -> retrieve.

Run this file directly to (re)build the vector store:
    python -m src.rag
"""

import json
import logging

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

logger = logging.getLogger(__name__)

# File written next to the FAISS index that records which embedding model built it.
INDEX_INFO_FILE = config.VECTOR_STORE_DIR / "index_info.json"

# The vector store is loaded once and reused (loading the embedding model is slow).
_vector_store = None


class KnowledgeBaseError(Exception):
    """Raised when the knowledge base or vector store cannot be used."""


def get_embeddings():
    """Return the embedding model chosen in .env (EMBEDDING_PROVIDER)."""
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


def load_documents():
    """Load every .txt file in data/knowledge_base as a LangChain Document."""
    if not config.KNOWLEDGE_BASE_DIR.exists():
        raise KnowledgeBaseError(
            f"Knowledge base folder not found: {config.KNOWLEDGE_BASE_DIR}"
        )

    files = sorted(config.KNOWLEDGE_BASE_DIR.glob("*.txt"))
    if not files:
        raise KnowledgeBaseError(
            f"No .txt files found in {config.KNOWLEDGE_BASE_DIR}"
        )

    documents = []
    for path in files:
        # encoding="utf-8" matters on Windows, where the default encoding differs.
        docs = TextLoader(str(path), encoding="utf-8").load()
        for doc in docs:
            doc.metadata["source"] = path.name  # keep just the file name
        documents.extend(docs)
    return documents


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


def build_vector_store():
    """Build the FAISS index from scratch and save it to vector_store/."""
    global _vector_store

    documents = load_documents()
    chunks = split_documents(documents)
    logger.info("Embedding %d chunks from %d documents...", len(chunks), len(documents))

    try:
        store = FAISS.from_documents(chunks, get_embeddings())
        config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(config.VECTOR_STORE_DIR))
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(f"Could not build the vector store: {exc}") from exc

    INDEX_INFO_FILE.write_text(
        json.dumps(
            {
                "embedding_model": _embedding_model_name(),
                "documents": len(documents),
                "chunks": len(chunks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _vector_store = store
    return store


def _index_is_usable() -> bool:
    """The saved index exists and was built with the current embedding model."""
    if not (config.VECTOR_STORE_DIR / "index.faiss").exists():
        return False
    try:
        info = json.loads(INDEX_INFO_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return info.get("embedding_model") == _embedding_model_name()


def get_vector_store():
    """Load the FAISS index from disk, building it first if needed."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if not _index_is_usable():
        logger.info("No usable vector store found, building a new one.")
        return build_vector_store()

    try:
        # allow_dangerous_deserialization: FAISS metadata is stored with pickle.
        # Safe here because we only ever load an index this app created itself.
        _vector_store = FAISS.load_local(
            str(config.VECTOR_STORE_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        logger.warning("Saved index could not be loaded (%s). Rebuilding.", exc)
        return build_vector_store()
    return _vector_store


def retrieve(query: str, k: int = config.TOP_K):
    """Return the k most relevant chunks for a query.

    Each result is a dict: {"source", "chunk", "score", "content"}.
    score is a distance: lower means more similar.
    """
    if not query or not query.strip():
        return []
    try:
        store = get_vector_store()
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
    # Manual check: build the index and run one test query.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_vector_store()
    info = json.loads(INDEX_INFO_FILE.read_text(encoding="utf-8"))
    print(f"\nVector store built: {info['documents']} documents, {info['chunks']} chunks")
    print(f"Embedding model: {info['embedding_model']}\n")

    question = "What is your refund policy?"
    print(f"Test query: {question}")
    for hit in retrieve(question, k=3):
        preview = hit["content"].replace("\n", " ")[:90]
        print(f"  {hit['source']} #{hit['chunk']} (distance {hit['score']}): {preview}...")
