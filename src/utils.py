"""Shared utilities: paths, text cleaning, JSONL I/O, GCP credentials."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import config

ROOT = config.PROJECT_ROOT


def ensure_src_on_path() -> None:
    """Allow running scripts as `python src/foo.py` from project root."""
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def resolve_path(relative: str | Path) -> Path:
    path = Path(relative)
    if not path.is_absolute():
        path = ROOT / path
    return path


def doc_id_from_filename(filename: str) -> str:
    """Stable doc_id from a PDF filename."""
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^\w]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "document"


def clean_text(text: str) -> str:
    """Normalize whitespace and fix common PDF line-break artifacts."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def split_words(text: str) -> list[str]:
    return text.split()


def word_count(text: str) -> int:
    return len(split_words(text))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = resolve_path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_gcp_project_id() -> str:
    if config.GCP_PROJECT_ID:
        return config.GCP_PROJECT_ID
    key_path = resolve_path(config.GCP_KEY_PATH)
    with key_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["project_id"]


def setup_gcp_credentials() -> str:
    """Set GOOGLE_APPLICATION_CREDENTIALS and return project_id."""
    key_path = resolve_path(config.GCP_KEY_PATH)
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(key_path))
    return load_gcp_project_id()


def iter_word_windows(
    words: list[str], size: int, overlap: int
) -> Iterator[tuple[int, list[str]]]:
    """Yield (start_index, word_slice) for sliding windows."""
    if not words:
        return
    if size <= 0:
        raise ValueError("chunk size must be positive")
    step = max(1, size - overlap)
    start = 0
    while start < len(words):
        yield start, words[start : start + size]
        if start + size >= len(words):
            break
        start += step


def metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma accepts str, int, float, bool metadata values only."""
    out: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = str(value)
    return out
