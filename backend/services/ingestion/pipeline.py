# ============================================================
# services/ingestion/pipeline.py - Full Ingestion Orchestrator
# ============================================================
#
# WHAT THIS FILE DOES:
# --------------------
# This is the CONDUCTOR of the ingestion orchestra. When a user 
# uploads a PDF, this file coordinates all the steps:
#
#   PDF uploaded → Parse PDF → Create chunks → Extract propositions
#                → Embed everything → Store in FAISS → Save to disk
#
# Each step is handled by a specialist module (pdf_parser, chunker, 
# proposition_extractor, embedder, faiss_store). This file just 
# calls them in the right order and handles errors.
#
# ASYNC BACKGROUND PROCESSING:
# ----------------------------
# The ingestion pipeline runs as a BACKGROUND TASK. This means:
# 1. User uploads PDF
# 2. API responds immediately with "got it, processing..."
# 3. Pipeline runs in the background (takes 2-5 minutes)
# 4. User polls /documents/{id}/status to check progress
#
# This is how production systems work. The alternative (making the 
# user wait at a blank screen for 5 minutes) is terrible UX.
# ============================================================

import json
import traceback
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.document import Document, ContextChunk, Proposition
from models.common import DocumentStatus
from config import settings

from services.ingestion.pdf_parser import parse_pdf, extract_title_from_pdf
from services.ingestion.chunker import create_chunks
from services.ingestion.proposition_extractor import extract_propositions_from_chunks
from services.retrieval.embedder import get_embedder
from services.retrieval.faiss_store import get_proposition_store, get_chunk_store
from services.retrieval.bm25_store import get_bm25_store


# ============================================================
# IN-MEMORY DOCUMENT REGISTRY
# ============================================================
# We track all documents and their status in a simple dict.
# For a production system, this would be a database.
# For a portfolio project, a dict + JSON file is sufficient.
#
# Key: document ID (string)
# Value: Document object (Pydantic model)
# ============================================================
_documents: dict[str, Document] = {}

# Also store propositions and chunks for display/lookup
_all_propositions: dict[str, Proposition] = {}  # key: prop ID
_all_chunks: dict[str, ContextChunk] = {}  # key: chunk ID


def _save_state() -> None:
    """
    Save the current state (documents, propositions, chunks) to JSON files.
    This is our "poor man's database" - it works for demo scale.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save document metadata
    docs_data = {doc_id: doc.model_dump(mode="json") for doc_id, doc in _documents.items()}
    with open(settings.documents_metadata_path, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=2, default=str)
    
    # Save propositions
    props_data = {pid: p.model_dump(mode="json") for pid, p in _all_propositions.items()}
    with open(settings.propositions_path, "w", encoding="utf-8") as f:
        json.dump(props_data, f, indent=2, default=str)
    
    # Save chunks
    chunks_data = {cid: c.model_dump(mode="json") for cid, c in _all_chunks.items()}
    with open(settings.chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, default=str)


def _load_state() -> None:
    """
    Load saved state from JSON files (called on startup).
    Also rebuilds the BM25 index from saved propositions.
    """
    global _documents, _all_propositions, _all_chunks
    
    # Load documents
    if settings.documents_metadata_path.exists():
        with open(settings.documents_metadata_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        _documents = {
            doc_id: Document(**doc_data) 
            for doc_id, doc_data in docs_data.items()
        }
        print(f" Loaded {len(_documents)} documents from disk")
    
    # Load propositions
    if settings.propositions_path.exists():
        with open(settings.propositions_path, "r", encoding="utf-8") as f:
            props_data = json.load(f)
        _all_propositions = {
            pid: Proposition(**pdata) 
            for pid, pdata in props_data.items()
        }
        print(f" Loaded {len(_all_propositions)} propositions from disk")
    
    # Load chunks
    if settings.chunks_path.exists():
        with open(settings.chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        _all_chunks = {
            cid: ContextChunk(**cdata) 
            for cid, cdata in chunks_data.items()
        }
        print(f" Loaded {len(_all_chunks)} chunks from disk")
    
    # Rebuild BM25 index from loaded propositions
    if _all_propositions:
        bm25_metadata = [
            {
                "id": prop.id,
                "text": prop.text,
                "doc_id": prop.doc_id,
                "chunk_id": prop.chunk_id,
                "section_type": prop.section_type.value,
                "doc_title": prop.doc_title,
            }
            for prop in _all_propositions.values()
        ]
        get_bm25_store().build(bm25_metadata)


def get_document(doc_id: str) -> Document | None:
    """Get a document by ID, or None if not found."""
    return _documents.get(doc_id)


def get_all_documents() -> list[Document]:
    """Get all documents, sorted by ingestion date (newest first)."""
    return sorted(_documents.values(), key=lambda d: d.date_ingested, reverse=True)


def get_proposition_by_id(prop_id: str) -> Proposition | None:
    """Get a proposition by ID."""
    return _all_propositions.get(prop_id)


def get_chunk_by_id(chunk_id: str) -> ContextChunk | None:
    """Get a context chunk by ID."""
    return _all_chunks.get(chunk_id)


async def run_ingestion_pipeline(doc_id: str, pdf_path: str) -> None:
    """
    Run the full ingestion pipeline for a document.
    
    This is called as a FastAPI BackgroundTask. It runs AFTER the 
    API response has been sent to the user.
    
    The pipeline steps:
    1. Parse PDF → structured text blocks
    2. Create chunks → ~400-token pieces with section tags
    3. Extract propositions → atomic facts from each chunk
    4. Embed chunks → store in FAISS chunk index
    5. Embed propositions → store in FAISS proposition index
    6. Update BM25 → rebuild keyword index
    7. Save everything to disk
    
    If any step fails, the document's status is set to FAILED with 
    the error message, so the user can see what went wrong.
    """
    doc = _documents.get(doc_id)
    if not doc:
        print(f" Document {doc_id} not found in registry!")
        return
    
    try:
        # ---- Update status ----
        doc.status = DocumentStatus.PROCESSING
        print(f"\n{'='*60}")
        print(f" Starting ingestion for: {doc.title}")
        print(f"{'='*60}")
        
        # ---- Step 1: Parse PDF ----
        print(f"\n Step 1/6: Parsing PDF...")
        text_blocks = parse_pdf(pdf_path)
        
        if not text_blocks:
            raise ValueError("No text could be extracted from this PDF. "
                           "It might be a scanned image (not searchable text).")
        
        # ---- Step 2: Create chunks ----
        print(f"\n  Step 2/6: Creating chunks...")
        chunks = create_chunks(text_blocks, doc_id=doc.id)
        doc.num_chunks = len(chunks)
        
        # Store chunks in our registry
        for chunk in chunks:
            _all_chunks[chunk.id] = chunk
        
        # ---- Step 3: Extract propositions ----
        print(f"\n Step 3/6: Extracting propositions (this takes a while)...")
        propositions = extract_propositions_from_chunks(chunks, doc_title=doc.title)
        doc.num_propositions = len(propositions)
        
        # Store propositions in our registry
        for prop in propositions:
            _all_propositions[prop.id] = prop
        
        # ---- Step 4: Embed chunks ----
        print(f"\n Step 4/6: Embedding {len(chunks)} chunks...")
        embedder = get_embedder()
        chunk_texts = [c.text for c in chunks]
        chunk_vectors = embedder.embed_batch(chunk_texts)
        
        # Build metadata for FAISS
        chunk_metadata = [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "section_type": chunk.section_type.value,
                "position": chunk.position,
                "token_count": chunk.token_count,
                "doc_title": doc.title,
            }
            for chunk in chunks
        ]
        
        # Add to FAISS chunk index
        chunk_store = get_chunk_store()
        chunk_store.add(chunk_vectors, chunk_metadata)
        
        # ---- Step 5: Embed propositions ----
        print(f"\n Step 5/6: Embedding {len(propositions)} propositions...")
        prop_texts = [p.text for p in propositions]
        prop_vectors = embedder.embed_batch(prop_texts)
        
        # Build metadata for FAISS
        prop_metadata = [
            {
                "id": prop.id,
                "doc_id": prop.doc_id,
                "chunk_id": prop.chunk_id,
                "text": prop.text,
                "section_type": prop.section_type.value,
                "doc_title": prop.doc_title,
            }
            for prop in propositions
        ]
        
        # Add to FAISS proposition index
        prop_store = get_proposition_store()
        prop_store.add(prop_vectors, prop_metadata)
        
        # ---- Step 6: Update BM25 + Save ----
        print(f"\n Step 6/6: Updating BM25 index and saving...")
        
        # Rebuild BM25 with ALL propositions (existing + new)
        all_prop_metadata = [
            {
                "id": prop.id,
                "text": prop.text,
                "doc_id": prop.doc_id,
                "chunk_id": prop.chunk_id,
                "section_type": prop.section_type.value,
                "doc_title": prop.doc_title,
            }
            for prop in _all_propositions.values()
        ]
        get_bm25_store().build(all_prop_metadata)
        
        # Save FAISS indexes to disk
        prop_store.save()
        chunk_store.save()
        
        # Save JSON state
        doc.status = DocumentStatus.COMPLETED
        doc.date_completed = datetime.utcnow()
        _save_state()
        
        print(f"\n{'='*60}")
        print(f" Ingestion complete for: {doc.title}")
        print(f"   Chunks: {doc.num_chunks}")
        print(f"   Propositions: {doc.num_propositions}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        # If anything goes wrong, mark the document as failed.
        # The error message is stored so the user can see what happened.
        doc.status = DocumentStatus.FAILED
        doc.error_message = str(e)
        _save_state()
        
        print(f"\n Ingestion FAILED for {doc.title}:")
        print(f"   {e}")
        traceback.print_exc()


def create_document_record(filename: str, pdf_path: str) -> Document:
    """
    Create a new Document record and add it to the registry.
    
    This is called BEFORE the ingestion pipeline starts.
    The document starts with status=PENDING.
    
    Args:
        filename: Original filename of the uploaded PDF.
        pdf_path: Where the PDF is saved on disk.
    
    Returns:
        The new Document object (with a generated ID).
    """
    # Try to extract the title from the PDF
    title = extract_title_from_pdf(pdf_path)
    
    doc = Document(
        title=title,
        source_filename=filename,
        status=DocumentStatus.PENDING
    )
    
    _documents[doc.id] = doc
    _save_state()
    
    print(f" Created document record: {doc.id} ({doc.title})")
    return doc


# Initialize state on module import
_load_state()
