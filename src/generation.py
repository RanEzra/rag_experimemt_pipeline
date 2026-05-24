"""Generate answers from retrieved context using Gemini."""

from __future__ import annotations

import re

from google.genai import types

import config
from src.embeddings import get_genai_client

CHUNK_ID_PATTERN = re.compile(r"\[([a-z0-9_]+_chunk_\d{4})\]", re.IGNORECASE)

SYSTEM_INSTRUCTION = """You are a question-answering assistant over CyberWell research PDFs.
Answer using ONLY the provided context.
If the answer is not in the context, say exactly: "The information was not found in the provided context."
Do not use outside knowledge.

Rules:
- Include exact numbers, dates, and percentages from the context when the question asks for them.
- For yes/no or "does the report claim" questions, state clearly yes or no and support with evidence.
- Cite every factual claim with the chunk_id in square brackets, e.g. [doc_name_chunk_0001].
- Prefer the chunk that directly contains the statistic or definition asked about."""


def format_context(retrieved_chunks: list[dict]) -> str:
    parts: list[str] = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page") or meta.get("page_start", "")
        page_info = f", page: {page}" if page else ""
        header = f"[chunk_id: {chunk['chunk_id']}] (source: {source}{page_info})"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "The information was not found in the provided context."

    client = get_genai_client()
    context = format_context(retrieved_chunks)
    prompt = f"""Question:
{question}

Context:
{context}

Answer:"""

    response = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    return (response.text or "").strip()


def extract_cited_chunk_ids(
    answer: str, retrieved_chunks: list[dict]
) -> list[str]:
    """Extract chunk IDs cited in the answer; fallback to all retrieved if none."""
    cited = CHUNK_ID_PATTERN.findall(answer)
    valid_ids = {c["chunk_id"] for c in retrieved_chunks}
    unique = []
    for cid in cited:
        if cid in valid_ids and cid not in unique:
            unique.append(cid)
    if unique:
        return unique
    return [c["chunk_id"] for c in retrieved_chunks]
