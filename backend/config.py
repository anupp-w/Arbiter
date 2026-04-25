# ============================================================
# config.py - Arbiter Configuration
# ============================================================
# 
# WHAT THIS FILE DOES:
# --------------------
# Think of this like a "settings panel" for your entire app.
# Instead of hardcoding values like API keys or model names 
# scattered across 20 files, we define them ALL here in one place.
#
# HOW IT WORKS:
# -------------
# Pydantic's BaseSettings class does something magical:
# it automatically reads your .env file and fills in the values.
# So if .env has GROQ_API_KEY=gsk_abc123, then settings.groq_api_key
# will equal "gsk_abc123" - no manual parsing needed.
#
# WHY THIS MATTERS:
# -----------------
# 1. ONE place to change any setting (not grep through 20 files)
# 2. Type safety - if you set a number field to "hello", it yells
# 3. Defaults - sensible defaults so the app works out of the box
# 4. Secrets stay in .env, never in code
# ============================================================

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """
    All configuration for Arbiter lives here.
    
    Each field maps to an environment variable in .env.
    For example:
        groq_api_key  →  reads GROQ_API_KEY from .env
        groq_model    →  reads GROQ_MODEL from .env
    
    The 'Field(default=...)' part means: "if the .env doesn't have 
    this variable, use this default value instead."
    """
    
    # ---- LLM (Groq) ----
    # Groq gives us access to Llama 3.3 70B for free.
    # We use it for 3 things: proposition extraction, contradiction 
    # detection, and answer generation.
    groq_api_key: str = Field(
        default="",
        description="Your Groq API key from https://console.groq.com/keys"
    )
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Which LLM to use on Groq. Llama 3.1 8B has higher rate limits."
    )
    
    # ---- Embeddings ----
    # An embedding model converts text into a list of numbers (a "vector")
    # that captures the MEANING of the text. Similar meanings = similar vectors.
    # bge-small is 384 dimensions - small enough to run on CPU, good enough
    # to be near the top of the MTEB benchmark.
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local embedding model. Runs on your GPU/CPU, no API needed."
    )
    embedding_dimension: int = Field(
        default=384,
        description="Number of dimensions in each embedding vector. Must match the model."
    )
    
    # ---- Reranker ----
    # After initial retrieval finds ~15 candidates, the reranker scores each
    # one more carefully. Think of retrieval as "casting a wide net" and 
    # reranking as "picking the best fish from the net."
    # L-12 (12 layers) is more accurate than L-6 (6 layers) - worth it 
    # for a portfolio project.
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        description="Local cross-encoder reranker. More accurate than L-6 variant."
    )
    
    # ---- Retrieval Parameters ----
    # These control how many results we fetch and keep at each stage.
    # Think of it like a funnel:
    #   Stage 1 (dense + sparse): Cast wide net → top_k_retrieval candidates
    #   Stage 2 (reranker): Score carefully → top_k_rerank winners
    top_k_retrieval: int = Field(
        default=10,
        description="How many candidates to fetch from each retrieval method."
    )
    top_k_rerank: int = Field(
        default=6,
        description="How many final results to keep after reranking."
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant. Higher = more equal weighting. 60 is standard."
    )
    
    # ---- Chunking Parameters ----
    # When we split a PDF into pieces, we need to decide how big each piece is.
    # Too small = loses context. Too big = dilutes the search signal.
    # 400 tokens is the sweet spot for academic papers.
    chunk_size_tokens: int = Field(
        default=400,
        description="Target size for each text chunk in tokens."
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        description="How many tokens to overlap between consecutive chunks."
    )
    
    # ---- File Paths ----
    # Where things are stored on disk.
    # Path() creates OS-independent file paths (works on Windows AND Linux).
    base_dir: Path = Field(
        default=Path(__file__).parent,
        description="Root directory of the backend."
    )
    
    @property
    def data_dir(self) -> Path:
        """Where all persistent data lives."""
        return self.base_dir / "data"
    
    @property
    def faiss_dir(self) -> Path:
        """Where FAISS index files are saved."""
        return self.data_dir / "faiss_indexes"
    
    @property
    def documents_dir(self) -> Path:
        """Where uploaded PDFs are stored."""
        return self.data_dir / "documents"
    
    @property
    def propositions_path(self) -> Path:
        """JSON file storing all extracted propositions (for BM25 + display)."""
        return self.data_dir / "propositions.json"

    @property
    def chunks_path(self) -> Path:
        """JSON file storing all context chunks (for display in answers)."""
        return self.data_dir / "chunks.json"

    @property 
    def documents_metadata_path(self) -> Path:
        """JSON file storing document metadata (title, date, status, etc.)."""
        return self.data_dir / "documents_metadata.json"

    # ---- FastAPI ----
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    class Config:
        # Tell Pydantic to read from .env file
        env_file = ".env"
        # Make field names case-insensitive 
        # (GROQ_API_KEY in .env maps to groq_api_key in Python)
        env_file_encoding = "utf-8"


# ============================================================
# THE SINGLETON PATTERN
# ============================================================
# We create ONE Settings object here, and every file that needs 
# settings imports this same object. This way:
#   - The .env file is read exactly once
#   - All files share the same configuration
#   - Changing a setting changes it everywhere
#
# Usage in any file:
#   from config import settings
#   print(settings.groq_api_key)
# ============================================================
settings = Settings()
