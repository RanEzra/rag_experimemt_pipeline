"""Load PDF documents from raw data directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

from src.utils import clean_text, doc_id_from_filename, resolve_path


def _extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text per page from a PDF."""
    pages: list[dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = clean_text(page.get_text("text"))
            if text:
                pages.append({"page": page_num + 1, "text": text})
    finally:
        doc.close()
    return pages


def load_documents(raw_dir: str | Path) -> list[dict[str, Any]]:
    """
    Load all PDFs from raw_dir, one document per file (pages merged).

    Returns:
        {"doc_id": str, "text": str, "metadata": {"source": str, "page_start": int, ...}}
    """
    raw_path = resolve_path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    pdf_files = sorted(raw_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {raw_path}")

    documents: list[dict[str, Any]] = []
    for pdf_path in pdf_files:
        base_doc_id = doc_id_from_filename(pdf_path.name)
        pages = _extract_pdf_pages(pdf_path)
        if not pages:
            continue
        parts: list[str] = []
        for p in pages:
            parts.append(f"\n\n--- Page {p['page']} ---\n\n{p['text']}")
        full_text = clean_text("".join(parts))
        documents.append(
            {
                "doc_id": base_doc_id,
                "text": full_text,
                "metadata": {
                    "source": pdf_path.name,
                    "page_start": pages[0]["page"],
                    "page_end": pages[-1]["page"],
                    "num_pages": len(pages),
                },
            }
        )
    return documents
