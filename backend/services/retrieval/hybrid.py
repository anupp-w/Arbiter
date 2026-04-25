# ============================================================
# services/retrieval/hybrid.py - Hybrid Retrieval + RRF Fusion
# ============================================================
#
# WHAT IS HYBRID RETRIEVAL?
# -------------------------
# Instead of using just ONE search method, we use THREE at the same 
# time and combine their results. This is like asking three different 
# experts for their opinion, then taking a vote.
#
# The three search methods:
# 1. DENSE (FAISS propositions) - finds by MEANING
# 2. DENSE (FAISS chunks) - finds by CONTEXT
# 3. SPARSE (BM25) - finds by KEYWORDS
#
# WHAT IS RRF (Reciprocal Rank Fusion)?
# --------------------------------------
# RRF is how we combine the three result lists into one.
#
# The problem: each method returns different scores on different scales.
# FAISS scores are 0-1, BM25 scores can be 0-50+.
# We can't just add them - that's like adding meters and pounds.
#
# RRF solution: ignore the scores, use only the RANK (position).
# The formula for each result: score = 1 / (k + rank)
#
# Example with k=60:
#   Result appears at rank 1 → 1/(60+1) = 0.0164
#   Result appears at rank 5 → 1/(60+5) = 0.0154
#   Result appears at rank 10 → 1/(60+10) = 0.0143
#
# If a result appears in MULTIPLE lists, we SUM the scores:
#   "BERT achieves 93.5%" appears at:
#     Rank 2 in FAISS propositions → 0.0161
#     Rank 1 in BM25 → 0.0164
#     Not found in FAISS chunks → 0.0000
#     TOTAL RRF score = 0.0325  ← this result gets boosted!
#
# Results that appear in multiple lists get HIGHER scores.
# This is the "wisdom of crowds" effect - agreement = confidence.
#
# WHY k=60?
# ---------
# k=60 is the standard value from the original RRF paper.
# It makes the scores decrease slowly with rank, so even 
# rank 10 results still contribute meaningfully.
# Lower k = rank matters more. Higher k = rank matters less.
# 60 is well-tested; no reason to change it for a portfolio project.
# ============================================================

import asyncio
import numpy as np
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
from services.retrieval.embedder import get_embedder
from services.retrieval.faiss_store import get_proposition_store, get_chunk_store
from services.retrieval.bm25_store import get_bm25_store


def _rrf_fuse(
    result_lists: list[list[dict]], 
    k: int = None
) -> list[dict]:
    """
    Fuse multiple result lists using Reciprocal Rank Fusion.
    
    Args:
        result_lists: List of result lists from different retrieval methods.
                     Each result dict must have an "id" key.
        k: RRF constant. Higher = more equal weighting across ranks.
    
    Returns:
        A single fused list, sorted by combined RRF score (descending).
        Each result dict has an added "rrf_score" field.
    
    Example:
        >>> list1 = [{"id": "a", "text": "..."}, {"id": "b", "text": "..."}]
        >>> list2 = [{"id": "b", "text": "..."}, {"id": "c", "text": "..."}]
        >>> fused = _rrf_fuse([list1, list2])
        >>> # "b" appears in both lists → highest RRF score
        >>> print(fused[0]["id"])  # "b"
    """
    k = k or settings.rrf_k  # Default from config (60)
    
    # Dictionary to accumulate RRF scores by result ID
    # Also stores the best metadata we've seen for each ID
    fused: dict[str, dict] = {}
    
    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            result_id = result.get("id", "")
            if not result_id:
                continue
            
            # Calculate RRF contribution for this rank
            rrf_contribution = 1.0 / (k + rank)
            
            if result_id in fused:
                # Already seen this result from another method → add to its score
                fused[result_id]["rrf_score"] += rrf_contribution
            else:
                # First time seeing this result → create entry
                entry = dict(result)  # Copy the metadata
                entry["rrf_score"] = rrf_contribution
                fused[result_id] = entry
    
    # Sort by RRF score (highest first)
    sorted_results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    
    return sorted_results


async def hybrid_retrieve(
    query: str, 
    top_k: int = None
) -> list[dict]:
    """
    Run hybrid retrieval: dense (propositions) + dense (chunks) + sparse (BM25).
    
    All three searches run CONCURRENTLY using asyncio.gather().
    This means they execute at the same time, not one after another.
    Total time = max(time_of_slowest_search), NOT sum(all_search_times).
    
    ASYNC EXPLAINED:
    ----------------
    Imagine ordering food at three different restaurants simultaneously
    instead of waiting for each one to finish before going to the next.
    
    Without async: Total wait = restaurant1 + restaurant2 + restaurant3
    With async:    Total wait = max(restaurant1, restaurant2, restaurant3)
    
    For our case, each FAISS search takes ~1ms and BM25 takes ~5ms,
    so async doesn't save much time. But it demonstrates the PATTERN
    that matters at production scale (where each search might take 100ms+).
    
    Args:
        query: The user's search query text.
        top_k: How many results to return from each method before fusion.
    
    Returns:
        Fused result list sorted by RRF score (descending).
        Each result has: id, text, doc_id, doc_title, section_type, rrf_score.
    """
    top_k = top_k or settings.top_k_retrieval
    
    # ---- Step 1: Embed the query ----
    # The same query text gets embedded ONCE and used for both FAISS searches.
    embedder = get_embedder()
    query_vector = embedder.embed(query)
    
    # ---- Step 2: Run all three searches concurrently ----
    # We use asyncio to run them in parallel. Since FAISS and BM25 
    # are CPU-bound (not I/O-bound), we wrap them in run_in_executor 
    # to avoid blocking the async event loop.
    
    loop = asyncio.get_event_loop()
    
    # Each search is wrapped in a lambda that we'll run in the thread pool
    prop_store = get_proposition_store()
    chunk_store = get_chunk_store()
    bm25_store = get_bm25_store()
    
    # Run all three searches in parallel using the thread pool
    # run_in_executor(None, func) runs func in a separate thread
    prop_results, chunk_results, bm25_results = await asyncio.gather(
        loop.run_in_executor(None, prop_store.search, query_vector, top_k),
        loop.run_in_executor(None, chunk_store.search, query_vector, top_k),
        loop.run_in_executor(None, bm25_store.search, query, top_k),
    )
    
    # ---- Step 3: RRF Fusion ----
    # Combine all three result lists into one, ranked by agreement
    fused = _rrf_fuse([prop_results, chunk_results, bm25_results])
    
    # Return top results
    return fused[:top_k * 2]  # Return 2x top_k for the reranker to filter down
