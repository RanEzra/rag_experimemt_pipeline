"""Chunk documents using configurable strategies."""

from __future__ import annotations

from typing import Any

import config
from src.utils import clean_text, iter_word_windows, split_words, word_count


def chunk_documents(
    documents: list[dict[str, Any]],
    strategy: str | None = None,
    size_words: int | None = None,
    overlap_words: int | None = None,
) -> list[dict[str, Any]]:
    """Chunk all documents using the configured strategy."""
    strategy = strategy or config.CHUNKING_STRATEGY
    size_words = size_words or config.CHUNK_SIZE_WORDS
    overlap_words = overlap_words or config.CHUNK_OVERLAP_WORDS

    if strategy == "fixed_words":
        return _chunk_all_fixed_words(documents, size_words, overlap_words)
    if strategy == "paragraph_window":
        return _chunk_all_paragraph_window(documents, size_words, overlap_words)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _make_chunk(
    doc: dict[str, Any],
    chunk_index: int,
    text: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc_id = doc["doc_id"]
    metadata = dict(doc.get("metadata", {}))
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "chunk_id": f"{doc_id}_chunk_{chunk_index:04d}",
        "doc_id": doc_id,
        "text": clean_text(text),
        "metadata": metadata,
    }


def _chunk_fixed_words(
    doc: dict[str, Any], size: int, overlap: int
) -> list[dict[str, Any]]:
    words = split_words(doc["text"])
    if not words:
        return []
    chunks: list[dict[str, Any]] = []
    for idx, (_, window) in enumerate(iter_word_windows(words, size, overlap)):
        if not window:
            continue
        chunks.append(_make_chunk(doc, idx, " ".join(window)))
    return chunks


def _chunk_all_fixed_words(
    documents: list[dict[str, Any]], size: int, overlap: int
) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        all_chunks.extend(_chunk_fixed_words(doc, size, overlap))
    return all_chunks


def _split_paragraphs(text: str) -> list[str]:
    import re

    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_paragraph_window(
    doc: dict[str, Any], size: int, overlap: int
) -> list[dict[str, Any]]:
    paragraphs = _split_paragraphs(doc["text"])
    if not paragraphs:
        return []

    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    start_idx = 0

    while start_idx < len(paragraphs):
        current_indices: list[int] = []
        current_words = 0
        idx = start_idx

        while idx < len(paragraphs):
            para_words = word_count(paragraphs[idx])
            if current_indices and current_words + para_words > size:
                break
            current_indices.append(idx)
            current_words += para_words
            idx += 1
            if current_words >= size:
                break

        if not current_indices:
            current_indices = [start_idx]
            idx = start_idx + 1

        current_text = "\n\n".join(paragraphs[i] for i in current_indices)
        chunks.append(_make_chunk(doc, chunk_index, current_text))
        chunk_index += 1

        if idx >= len(paragraphs):
            break

        overlap_indices: list[int] = []
        overlap_total = 0
        for pi in reversed(current_indices):
            overlap_indices.insert(0, pi)
            overlap_total += word_count(paragraphs[pi])
            if overlap_total >= overlap:
                break

        next_start = overlap_indices[0] if overlap_indices else idx
        if next_start <= start_idx:
            next_start = idx
        start_idx = next_start

    return chunks


def _chunk_all_paragraph_window(
    documents: list[dict[str, Any]], size: int, overlap: int
) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        all_chunks.extend(_chunk_paragraph_window(doc, size, overlap))
    return all_chunks
