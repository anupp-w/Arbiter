# ============================================================
# backend/main.py — FastAPI Application Entry Point
# ============================================================
#
# WHAT THIS FILE DOES:
# --------------------
# This is the "front door" of the FastAPI backend.
# It creates the app, configures it, and tells it which 
# routers (URL groups) to include.
#
# LIFESPAN (the startup/shutdown hook):
# ---------------------------------------
# FastAPI has a "lifespan" concept — code that runs when the 
# server STARTS and when it STOPS. We use this to:
#   ON STARTUP: Load the embedding model, reranker, FAISS indexes
#   ON SHUTDOWN: Save any final state (handled automatically)
#
# WHY LOAD MODELS ON STARTUP?
# ----------------------------
# Loading a model takes 2-5 seconds. If we loaded it on every 
# single request, the first request to each endpoint would be 
# 5 seconds slower than all others. 
#
# Instead, we load ONCE at startup. Then every request uses the 
# already-loaded model (fast, consistent latency).
#
# CORS (Cross-Origin Resource Sharing):
# --------------------------------------
# By default, browsers block JavaScript from calling APIs on 
# different domains/ports. Since our Streamlit frontend is on 
# port 8501 and FastAPI is on port 8000, we need to explicitly 
# ALLOW these cross-origin requests. That's what CORSMiddleware does.
#
# Running:
#   cd backend
#   uvicorn main:app --reload --port 8000
# ============================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code that runs when the server starts and stops.
    
    The code BEFORE 'yield' runs on startup.
    The code AFTER 'yield' runs on shutdown.
    
    We use this to pre-load all heavy models so they're ready 
    when the first request arrives.
    """
    # ============================================================
    # STARTUP
    # ============================================================
    print("\n" + "="*60)
    print("🚀 Starting Arbiter API Server")
    print("="*60)
    
    # Create all required directories
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    print("📁 Data directories created/verified")
    
    # Pre-load the embedding model (downloads from HuggingFace on first run)
    # This is the EXPENSIVE operation — takes 2-5 seconds.
    # But it only happens once — all subsequent requests are fast.
    print("\n📦 Loading models (this may take a moment on first run)...")
    try:
        from services.retrieval.embedder import get_embedder
        embedder = get_embedder()
        print("✅ Embedding model ready")
    except Exception as e:
        print(f"❌ Warning: Embedding model failed to load: {e}")
    
    # Pre-load the reranker
    try:
        from services.retrieval.reranker import get_reranker
        reranker = get_reranker()
        print("✅ Reranker ready")
    except Exception as e:
        print(f"❌ Warning: Reranker failed to load: {e}")
        
    # Setup Tracing (Phoenix)
    try:
        from services.tracing.setup import setup_tracing
        setup_tracing(app)
    except Exception as e:
        print(f"❌ Warning: Tracing setup failed: {e}")
        
    # Load FAISS indexes from disk (if previously saved)
    # The pipeline.py module already calls _load_state() on import,
    # which loads both FAISS indexes and rebuilds BM25.
    try:
        import services.ingestion.pipeline  # triggers _load_state()
        print("✅ Document index loaded")
    except Exception as e:
        print(f"❌ Warning: State loading failed: {e}")
    
    print("\n" + "="*60)
    print("✅ Arbiter is ready!")
    print(f"   API docs: http://localhost:{settings.api_port}/docs")
    print("="*60 + "\n")
    
    # yield hands control back to FastAPI.
    # The server runs until it's stopped (Ctrl+C).
    yield
    
    # ============================================================
    # SHUTDOWN (code after yield runs when server stops)
    # ============================================================
    print("\n👋 Arbiter shutting down...")


# ============================================================
# CREATE THE FASTAPI APP
# ============================================================
app = FastAPI(
    title="Arbiter — Multi-Document Research Synthesizer",
    description=(
        "An agentic RAG system with contradiction-aware retrieval. "
        "Upload research papers, ask questions, and get structured answers "
        "where every claim is labeled: Consensus, Disputed, or Single-Source."
    ),
    version="1.0.0",
    lifespan=lifespan,  # Register our startup/shutdown hook
    docs_url="/docs",   # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc"  # ReDoc UI at http://localhost:8000/redoc
)

# ============================================================
# CORS MIDDLEWARE
# ============================================================
# This allows the Streamlit frontend (port 8501) to call our API (port 8000).
# In production, you'd restrict origins to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins (fine for local dev)
    allow_credentials=True,
    allow_methods=["*"],       # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],       # Allow all headers
)

# ============================================================
# INCLUDE ROUTERS
# ============================================================
# Each router handles a group of related endpoints.
# By including them here, their routes become part of the main app.
from routers.documents import router as documents_router
from routers.query import router as query_router

app.include_router(documents_router)   # /documents, /documents/{id}/status
app.include_router(query_router)       # /query


# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.get("/")
async def root():
    """Health check and welcome message."""
    return {
        "name": "Arbiter API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "description": "Multi-document research synthesizer with contradiction detection"
    }


@app.get("/health")
async def health():
    """Health check endpoint for deployment monitoring."""
    from services.retrieval.faiss_store import get_proposition_store, get_chunk_store
    from services.ingestion.pipeline import get_all_documents
    
    prop_count = get_proposition_store().count
    chunk_count = get_chunk_store().count
    doc_count = len(get_all_documents())
    
    return {
        "status": "healthy",
        "documents_indexed": doc_count,
        "propositions_indexed": prop_count,
        "chunks_indexed": chunk_count
    }


# ============================================================
# RUN THE SERVER
# ============================================================
# This block runs when you execute: python main.py
# For production, use: uvicorn main:app --host 0.0.0.0 --port 8000
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True  # Auto-restart on file changes (great for development)
    )
