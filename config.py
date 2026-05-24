"""Central configuration for the RAG pipeline. Change values here to run experiments."""

from pathlib import Path

# --- Chunking ---
CHUNKING_STRATEGY = "paragraph_window"  # "fixed_words" | "paragraph_window"
CHUNK_SIZE_WORDS = 2000
CHUNK_OVERLAP_WORDS = 200

# --- Retrieval ---
TOP_K = 5
EVAL_HIT_K = 5  # standard metric for course report (Hit@5)

# --- Models (Vertex AI / Gemini) ---
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_BATCH_SIZE = 4  # keep each embed request under Vertex token limits

# --- GCP ---
GCP_KEY_PATH = "gcp_key.json"
GCP_LOCATION = "us-central1"
# Set to None to read project_id from the service account JSON.
GCP_PROJECT_ID: str | None = None

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = "data/raw"
PROCESSED_DOCS_PATH = "data/processed/documents.jsonl"
PROCESSED_CHUNKS_PATH = "data/processed/chunks.jsonl"
CHROMA_PERSIST_DIR = "data/processed/chroma"
CHROMA_COLLECTION = "cyberwell_rag"

# --- Build pipeline ---
REBUILD_FROM_RAW = True  # set False to reuse documents.jsonl / chunks.jsonl when re-indexing

# --- Future improvements (not implemented) ---
# USE_BM25 = False
# USE_RERANKING = False
