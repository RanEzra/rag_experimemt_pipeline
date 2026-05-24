"""Gemini embeddings via Vertex AI."""

from __future__ import annotations

import time
from functools import lru_cache

from google import genai
from google.genai import types

import config
from src.utils import setup_gcp_credentials


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    project_id = setup_gcp_credentials()
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=config.GCP_LOCATION,
    )


def embed_texts(
    texts: list[str],
    batch_size: int | None = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed a list of texts, batched with simple retry on rate limits."""
    if not texts:
        return []
    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
    client = get_genai_client()
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings = _embed_batch_with_retry(client, batch, task_type=task_type)
        all_embeddings.extend(embeddings)
    return all_embeddings


def embed_query(query: str) -> list[float]:
    result = embed_texts([query], task_type="RETRIEVAL_QUERY")
    return result[0]


def _embed_batch_with_retry(
    client: genai.Client,
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    max_retries: int = 5,
) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                ),
            )
            return [list(e.values) for e in response.embeddings]
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
    return []
