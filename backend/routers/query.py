# ============================================================
# routers/query.py — Query Processing Endpoint
# ============================================================
#
# THE FULL QUERY PIPELINE IN ONE PLACE:
# --------------------------------------
# This router handles POST /query. When a user asks a question,
# this function runs the ENTIRE pipeline:
#
#   1. Hybrid retrieval (FAISS + BM25, concurrent)
#   2. Cross-encoder reranking
#   3. Contradiction detection
#   4. Structured answer generation
#   5. Source verification (hallucination guard)
#   6. Claim status classification
#   7. Confidence calibration
#   8. Return structured result
#
# Total time: ~5-10 seconds on your machine.
# (Retrieval: ~1s, Reranking: ~0.5s, Contradiction: ~2s, Generation: ~3s)
# ============================================================

import time
import asyncio
from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.query import QueryRequest, QueryResult, RetrievedProposition
from config import settings

from services.retrieval.hybrid import hybrid_retrieve
from services.retrieval.reranker import get_reranker
from services.analysis.contradiction import detect_contradictions
from services.analysis.claim_classifier import classify_all_claims
from services.generation.answer_generator import generate_answer
from services.generation.post_processor import verify_sources, compute_confidence

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=dict)
async def process_query(request: QueryRequest):
    """
    Process a research query through the full Arbiter pipeline.
    
    This is the main endpoint. It runs:
    retrieval → reranking → contradiction detection → generation →
    source verification → claim classification → confidence calibration
    
    Request body:
        {"query_text": "How do scaling laws differ between papers?"}
    
    Response:
        Full QueryResult object with:
        - main_answer: prose response
        - claims: list of claims with Consensus/Disputed/Single-Source badges
        - contradictions: detected conflicts between papers
        - confidence: calibrated confidence breakdown
        - sources: which documents contributed
    """
    start_time = time.time()
    query = request.query_text.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    print(f"\n{'='*60}")
    print(f"🔍 Query: {query}")
    print(f"{'='*60}")
    
    # ---- Step 1: Hybrid Retrieval ----
    print(f"\n📡 Step 1: Hybrid retrieval...")
    try:
        # This runs FAISS dense + BM25 sparse + RRF fusion concurrently
        raw_results = await hybrid_retrieve(query, top_k=settings.top_k_retrieval)
        print(f"   Retrieved {len(raw_results)} candidates via hybrid retrieval")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    
    if not raw_results:
        # No results at all — index might be empty
        return QueryResult(
            query_text=query,
            main_answer="No documents have been indexed yet. Please upload some PDFs first.",
            claims=[],
            processing_time_seconds=time.time() - start_time
        ).model_dump(mode="json")
    
    # ---- Step 2: Cross-encoder Reranking ----
    print(f"\n🎯 Step 2: Reranking top {len(raw_results)} candidates...")
    
    # We run reranking in an executor because it's CPU-bound (not I/O-bound)
    # Running it in executor prevents it from blocking the async event loop
    loop = asyncio.get_event_loop()
    reranker = get_reranker()
    top_results = await loop.run_in_executor(
        None, 
        reranker.rerank, 
        query, 
        raw_results, 
        settings.top_k_rerank
    )
    
    print(f"   Reranked to top {len(top_results)}")
    
    # ---- Convert raw dicts to RetrievedProposition objects ----
    # The retrieval pipeline returns plain dicts (easier for FAISS/BM25).
    # We now convert to our typed Pydantic models for the rest of the pipeline.
    retrieved_props = []
    for r in top_results:
        prop = RetrievedProposition(
            proposition_id=r.get("id", ""),
            text=r.get("text", ""),
            doc_id=r.get("doc_id", ""),
            doc_title=r.get("doc_title", "Unknown Paper"),
            chunk_id=r.get("chunk_id", ""),
            section_type=r.get("section_type", "other"),
            rrf_score=r.get("rrf_score", 0.0),
            reranker_score=r.get("reranker_score", 0.0)
        )
        retrieved_props.append(prop)
    
    # ---- Step 3: Contradiction Detection ----
    print(f"\n⚡ Step 3: Checking {len(retrieved_props)} propositions for contradictions...")
    
    contradictions = await loop.run_in_executor(
        None,
        detect_contradictions,
        retrieved_props
    )
    
    print(f"   Found {len(contradictions)} contradiction(s)")
    
    # ---- Step 4: Structured Generation ----
    print(f"\n✍️  Step 4: Generating structured answer...")
    
    prose_answer, raw_claims = await loop.run_in_executor(
        None,
        generate_answer,
        query,
        retrieved_props,
        contradictions
    )
    
    # ---- Step 5: Source Verification (Hallucination Guard) ----
    print(f"\n🔒 Step 5: Verifying sources (hallucination guard)...")
    verified_claims = verify_sources(raw_claims, retrieved_props)
    
    # ---- Step 6: Claim Classification ----
    print(f"\n🏷️  Step 6: Classifying claims (Consensus/Disputed/Single-Source)...")
    classified_claims = classify_all_claims(verified_claims, retrieved_props, contradictions)
    
    # ---- Step 7: Confidence Calibration ----
    confidence = compute_confidence(
        claims=classified_claims,
        retrieved_propositions=retrieved_props,
        fallback_triggered=False
    )
    
    # ---- Assemble the result ----
    total_time = time.time() - start_time
    
    result = QueryResult(
        query_text=query,
        main_answer=prose_answer,
        claims=classified_claims,
        contradictions=contradictions,
        retrieved_propositions=retrieved_props,
        confidence=confidence,
        processing_time_seconds=round(total_time, 2)
    )
    
    print(f"\n✅ Query complete in {total_time:.1f}s")
    print(f"   Claims: {len(classified_claims)}, Contradictions: {len(contradictions)}")
    print(f"   Confidence: {confidence.overall:.2f}")
    
    return result.model_dump(mode="json")
