# ============================================================
# routers/documents.py - Document Upload & Status Endpoints
# ============================================================
#
# FastAPI ENDPOINTS EXPLAINED:
# ----------------------------
# An endpoint is a URL that your API listens to.
# When someone sends a request to that URL, your function runs.
#
# Example:
#   @app.post("/documents")
#   async def upload_doc(...)
# 
# This means: "when someone sends a POST request to /documents,
# run the upload_doc() function and return whatever it returns."
#
# THE TWO ENDPOINTS HERE:
# -----------------------
# 1. POST /documents - Upload a PDF for ingestion
#    - Accepts the PDF file
#    - Saves it to disk
#    - Creates a Document record (status: PENDING)
#    - Returns immediately with doc_id (doesn't wait for ingestion)
#    - Kicks off ingestion as a BACKGROUND TASK
#
# 2. GET /documents/{doc_id}/status - Check ingestion progress
#    - Returns the Document with its current status
#    - The UI polls this every 3 seconds after upload
#    - When status = COMPLETED, the UI stops polling
#
# 3. GET /documents - List all documents
#    - Returns all documents with their status
#    - Used by the sidebar to show what's been indexed
# ============================================================

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.document import Document
from models.common import DocumentStatus
from services.ingestion.pipeline import (
    create_document_record, 
    run_ingestion_pipeline,
    get_document,
    get_all_documents
)
from config import settings

# APIRouter is like a "mini app" - we define routes here, 
# then attach this router to the main FastAPI app in main.py.
# This keeps our code organized: documents routes in one file,
# query routes in another.
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a PDF document for ingestion.
    
    This endpoint does THREE things fast, then returns:
    1. Validates the file is a PDF
    2. Saves the PDF to disk
    3. Creates a Document record with status=PENDING
    4. Schedules the ingestion pipeline as a background task
    5. Returns immediately with the doc_id
    
    The client should then poll GET /documents/{doc_id}/status
    to track ingestion progress.
    
    Request: multipart/form-data with a "file" field (PDF)
    Response: {"doc_id": "...", "title": "...", "status": "pending"}
    """
    # ---- Validate file type ----
    # We only accept PDFs. Check both the filename extension AND the 
    # MIME type (content-type header) for safety.
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )
    
    # ---- Save to disk ----
    # Create the documents directory if it doesn't exist
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the save path
    # We use the original filename - if duplicates exist, overwrite is fine 
    # for a portfolio project (production would add a UUID suffix)
    save_path = settings.documents_dir / file.filename
    
    try:
        # Save the uploaded file to disk
        # shutil.copyfileobj copies the file content efficiently (streaming)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # ---- Create Document record ----
    doc = create_document_record(
        filename=file.filename,
        pdf_path=str(save_path)
    )
    
    # ---- Schedule background ingestion ----
    # BackgroundTasks.add_task() tells FastAPI:
    # "After you've sent the response to the client, run this function."
    # The function runs in the SAME process but after the HTTP response is sent.
    # This means the client gets a response in <1 second, even though 
    # ingestion takes 2-5 minutes.
    background_tasks.add_task(
        run_ingestion_pipeline,
        doc_id=doc.id,
        pdf_path=str(save_path)
    )
    
    return {
        "doc_id": doc.id,
        "title": doc.title,
        "status": doc.status.value,
        "message": (
            f"Document '{doc.title}' received and queued for processing. "
            f"Poll GET /documents/{doc.id}/status to track progress."
        )
    }


@router.get("/{doc_id}/status", response_model=dict)
async def get_document_status(doc_id: str):
    """
    Get the current status of a document.
    
    The UI polls this endpoint after uploading a document.
    
    Status values:
    - "pending"    → Queued, ingestion hasn't started yet
    - "processing" → Currently parsing, chunking, extracting propositions
    - "completed"  → Fully indexed and ready for queries
    - "failed"     → Something went wrong (error_message explains what)
    
    Response: Full Document object as JSON
    """
    doc = get_document(doc_id)
    
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found."
        )
    
    return {
        "doc_id": doc.id,
        "title": doc.title,
        "status": doc.status.value,
        "num_chunks": doc.num_chunks,
        "num_propositions": doc.num_propositions,
        "error_message": doc.error_message,
        "date_ingested": doc.date_ingested.isoformat() if doc.date_ingested else None,
        "date_completed": doc.date_completed.isoformat() if doc.date_completed else None,
    }


@router.get("", response_model=list)
async def list_documents():
    """
    List all documents in the system.
    
    Returns all documents sorted by ingestion date (newest first).
    Used by the sidebar to show what's been indexed.
    """
    docs = get_all_documents()
    return [
        {
            "doc_id": doc.id,
            "title": doc.title,
            "status": doc.status.value,
            "num_chunks": doc.num_chunks,
            "num_propositions": doc.num_propositions,
            "date_ingested": doc.date_ingested.isoformat() if doc.date_ingested else None,
        }
        for doc in docs
    ]
