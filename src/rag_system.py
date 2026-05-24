"""RAG system entry point with required answer() interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import ensure_src_on_path

ensure_src_on_path()

import config
from src.generation import extract_cited_chunk_ids, generate_answer
from src.retrieval import retrieve


def answer(question: str) -> dict:
    """
    Answer a question using retrieval-augmented generation.

    Returns:
        {
            "answer": str,
            "sources": list[str],
            "retrieved_chunks": list[dict],
        }
    """
    retrieved_chunks = retrieve(question, k=config.TOP_K)
    answer_text = generate_answer(question, retrieved_chunks)
    sources = extract_cited_chunk_ids(answer_text, retrieved_chunks)
    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python src/rag_system.py "Your question here"')
        sys.exit(1)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:])
    result = answer(question)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
