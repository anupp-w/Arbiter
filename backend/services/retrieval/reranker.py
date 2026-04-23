# ============================================================
# services/retrieval/reranker.py — Cross-Encoder Reranking
# ============================================================
#
# WHAT IS RERANKING?
# ------------------
# After hybrid retrieval gives us ~15 candidates, we need to pick 
# the BEST 6. The retrieval scores (FAISS cosine + BM25 + RRF) are 
# good but rough. The reranker is more accurate but slower.
#
# ANALOGY:
# --------
# Think of hiring:
#   1. Resume screening (retrieval) → quickly filters 1000 applicants to 15
#   2. Interview (reranking) → carefully evaluates 15 to pick the best 6
#
# You wouldn't interview 1000 people (too slow), and you wouldn't 
# hire from resumes alone (too inaccurate). The two-stage pipeline 
# gives you both speed AND accuracy.
#
# HOW A CROSS-ENCODER WORKS:
# --------------------------
# A regular embedding (bi-encoder) embeds the query and document 
# SEPARATELY, then compares vectors:
#   embed("question") → vec1
#   embed("document")  → vec2
#   score = cosine_similarity(vec1, vec2)
#
# A cross-encoder feeds query AND document TOGETHER into the model:
#   model("question [SEP] document") → score
#
# This is more accurate because the model can see both texts at once
# and find subtle relationships. But it's slower (can't pre-compute).
#
# That's why we use it ONLY on the top 15, not on all 5000 propositions.
#
# THE MODEL:
# ----------
# cross-encoder/ms-marco-MiniLM-L-12-v2
# - Trained on MS MARCO (Microsoft's search relevance dataset)
# - L-12 = 12 transformer layers (more accurate than L-6)
# - Runs locally, no API needed
# - Takes ~0.5s to score 15 pairs on your 1650Ti
# ============================================================

from sentence_transformers import CrossEncoder
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
import torch


class Reranker:
    """
    Cross-encoder reranker that scores (query, document) pairs.
    
    Usage:
        reranker = Reranker()
        scored = reranker.rerank(query="how accurate is BERT?", results=[...])
        # scored[0] is the most relevant result
    """
    
    def __init__(self):
        """
        Load the cross-encoder model.
        
        Like the embedding model, this downloads from HuggingFace on first 
        run (~130MB) and uses GPU if available.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📦 Loading reranker: {settings.reranker_model} (on {device})...")
        
        self.model = CrossEncoder(
            settings.reranker_model,
            max_length=512,  # Maximum input length in tokens
            device=device
        )
        
        print("✅ Reranker loaded!")
    
    def rerank(
        self, 
        query: str, 
        results: list[dict], 
        top_k: int = None
    ) -> list[dict]:
        """
        Rerank retrieval results using the cross-encoder.
        
        For each result, the model scores the (query, result_text) pair.
        Results are then sorted by this new score (best first).
        
        IMPORTANT: The reranker score is the MOST TRUSTWORTHY relevance signal.
        If the reranker gives a result a low score, it's probably not relevant,
        even if FAISS thought it was.
        
        Args:
            query: The user's original query text.
            results: List of result dicts from hybrid retrieval.
                    Each must have a "text" field.
            top_k: How many top results to return after reranking.
        
        Returns:
            The top_k results, sorted by reranker score (descending).
            Each result dict gets an added "reranker_score" field.
        """
        top_k = top_k or settings.top_k_rerank
        
        if not results:
            return []
        
        # ---- Build (query, document) pairs ----
        # The cross-encoder needs pairs of [query, document_text]
        pairs = [[query, r.get("text", "")] for r in results]
        
        # ---- Score all pairs ----
        # predict() returns an array of scores, one per pair.
        # On your 1650Ti: ~15 pairs takes ~0.3-0.5 seconds
        #
        # batch_size=16 means we process up to 16 pairs at once.
        # For 15 results, this means one batch — efficient.
        raw_scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False
        )
        
        # ---- Normalize scores to 0-1 range ----
        # Cross-encoder raw scores can be any float (negative to positive).
        # We use a sigmoid function to squish them into 0-1 range.
        # sigmoid(x) = 1 / (1 + e^(-x))
        # This makes scores interpretable: 0.8 = very relevant, 0.2 = not relevant.
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        normalized_scores = sigmoid(np.array(raw_scores))
        
        # ---- Attach scores and sort ----
        scored_results = []
        for result, score in zip(results, normalized_scores):
            result_copy = dict(result)
            result_copy["reranker_score"] = float(score)
            scored_results.append(result_copy)
        
        # Sort by reranker score (highest first)
        scored_results.sort(key=lambda x: x["reranker_score"], reverse=True)
        
        # ---- Log score distribution ----
        # A FLAT distribution (all scores close together) signals poor retrieval.
        # A STEEP distribution (one high, rest low) signals good retrieval.
        # We log this for observability.
        scores_array = [r["reranker_score"] for r in scored_results]
        if scores_array:
            score_spread = max(scores_array) - min(scores_array)
            print(f"   Reranker scores: max={max(scores_array):.3f}, "
                  f"min={min(scores_array):.3f}, spread={score_spread:.3f}")
            if score_spread < 0.1:
                print("   ⚠️  Low score spread — retrieval quality may be poor")
        
        return scored_results[:top_k]


# ============================================================
# SINGLETON
# ============================================================
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Get the singleton Reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
