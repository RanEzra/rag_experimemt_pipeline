"""Build ChromaDB index from raw PDFs."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import ensure_src_on_path

ensure_src_on_path()

import chromadb
from tqdm import tqdm

import config
from src.chunking import chunk_documents
from src.embeddings import embed_texts
from src.loaders import load_documents
from src.utils import metadata_for_chroma, read_jsonl, resolve_path, write_jsonl


def build_documents() -> list[dict]:
    docs_path = resolve_path(config.PROCESSED_DOCS_PATH)
    if not config.REBUILD_FROM_RAW and docs_path.exists():
        print(f"Loading cached documents from {docs_path}")
        return read_jsonl(docs_path)

    print(f"Loading PDFs from {config.RAW_DATA_DIR}")
    documents = load_documents(config.RAW_DATA_DIR)
    write_jsonl(docs_path, documents)
    print(f"Wrote {len(documents)} documents to {docs_path}")
    return documents


def build_chunks(documents: list[dict]) -> list[dict]:
    chunks_path = resolve_path(config.PROCESSED_CHUNKS_PATH)
    if not config.REBUILD_FROM_RAW and chunks_path.exists():
        print(f"Loading cached chunks from {chunks_path}")
        return read_jsonl(chunks_path)

    print(
        f"Chunking with strategy={config.CHUNKING_STRATEGY}, "
        f"size={config.CHUNK_SIZE_WORDS}, overlap={config.CHUNK_OVERLAP_WORDS}"
    )
    chunks = chunk_documents(documents)
    write_jsonl(chunks_path, chunks)
    print(f"Wrote {len(chunks)} chunks to {chunks_path}")
    return chunks


def build_chroma_index(chunks: list[dict]) -> None:
    persist_dir = resolve_path(config.CHROMA_PERSIST_DIR)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        metadata_for_chroma(
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                **c.get("metadata", {}),
            }
        )
        for c in chunks
    ]

    print(f"Embedding {len(texts)} chunks...")
    batch_size = config.EMBEDDING_BATCH_SIZE
    for start in tqdm(range(0, len(texts), batch_size)):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]
        batch_embeddings = embed_texts(batch_texts)
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    print(f"Indexed {collection.count()} chunks in {persist_dir}")


def main() -> None:
    documents = build_documents()
    chunks = build_chunks(documents)
    if not chunks:
        print("No chunks produced. Check raw PDFs and chunking settings.")
        sys.exit(1)
    build_chroma_index(chunks)
    print("Index build complete.")


if __name__ == "__main__":
    main()
