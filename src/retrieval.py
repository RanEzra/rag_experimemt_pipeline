"""Retrieve relevant chunks from ChromaDB."""

from __future__ import annotations

from functools import lru_cache

import chromadb

import config
from src.embeddings import embed_query
from src.utils import ensure_src_on_path, resolve_path

ensure_src_on_path()


@lru_cache(maxsize=1)
def get_collection():
    persist_dir = resolve_path(config.CHROMA_PERSIST_DIR)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(name=config.CHROMA_COLLECTION)

   
def distance_to_score(distance: float) -> float:
    """Convert Chroma distance to a similarity score in (0, 1]."""
    return 1.0 / (1.0 + distance)


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """
    Retrieve top-k chunks for a query.

    Returns list of:
        {"chunk_id", "text", "score", "metadata"}
    """
    k = k or config.TOP_K
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict] = []
    if not results["ids"] or not results["ids"][0]:
        return chunks

    for i, chunk_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i] if results["distances"] else 0.0
        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
        text = results["documents"][0][i] if results["documents"] else ""
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "score": distance_to_score(distance),
                "metadata": metadata or {},
            }
        )
    return chunks
