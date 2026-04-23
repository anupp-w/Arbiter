# ============================================================
# services/retrieval/bm25_store.py — BM25 Sparse Retrieval
# ============================================================
#
# WHAT IS BM25?
# -------------
# BM25 is Google's ORIGINAL search algorithm (before they added AI).
# It's a keyword-matching algorithm — it finds documents that contain 
# the same WORDS as your query.
#
# ANALOGY:
# --------
# Dense retrieval (FAISS) = finding books by MEANING
#   Query: "how well does the model perform?"
#   Finds: "BERT achieves 93.5% accuracy" (different words, same meaning)
#
# Sparse retrieval (BM25) = finding books by WORDS
#   Query: "BERT accuracy SQuAD"
#   Finds: documents containing "BERT", "accuracy", "SQuAD" literally
#
# WHY USE BOTH?
# -------------
# Dense retrieval is smart but sometimes misses EXACT keyword matches.
# BM25 is "dumb" but catches exact terms that vectors might miss.
# Together, they cover each other's blind spots. This combination 
# is called HYBRID RETRIEVAL.
#
# Example where BM25 shines and vectors fail:
#   Query: "MMLU benchmark"
#   Dense retrieval might return results about "model evaluation" in general
#   BM25 returns results that literally contain "MMLU" — exactly what we want
#
# Example where vectors shine and BM25 fails:
#   Query: "how large is the model?"
#   BM25 looks for the word "large" literally
#   Dense retrieval understands you're asking about model SIZE/parameters
#
# HOW BM25 WORKS (simplified):
# ----------------------------
# 1. Count how often each query word appears in each document (TF)
# 2. Penalize common words like "the", "is" (IDF)
# 3. Score = TF × IDF (approximately)
# 4. Return documents with highest scores
#
# It's the same concept as TF-IDF (which you used in JobGenie!),
# but with better math for handling document length differences.
# ============================================================

from rank_bm25 import BM25Okapi
from typing import Optional
import re


class BM25Store:
    """
    In-memory BM25 index over proposition texts.
    
    We only index PROPOSITIONS (not chunks) with BM25 because:
    - Propositions are short and keyword-dense
    - Chunks have too much noise for keyword matching
    - Memory is limited — proposition texts are much smaller
    
    The index is rebuilt from scratch each time the app starts 
    (loaded from the saved propositions JSON). This is fine for 
    demo scale — it takes < 1 second for ~5000 propositions.
    """
    
    def __init__(self):
        """Initialize an empty BM25 store."""
        self.index: Optional[BM25Okapi] = None
        self.documents: list[dict] = []  # [{id, text, doc_id, ...}, ...]
        self._tokenized_docs: list[list[str]] = []
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Split text into tokens (words) for BM25.
        
        This is MUCH simpler than LLM tokenization. We just:
        1. Lowercase everything
        2. Split on non-alphanumeric characters
        3. Remove very short tokens (1-2 chars)
        
        Example:
            "BERT achieves 93.5% on SQuAD" 
            → ["bert", "achieves", "935", "squad"]
        """
        # Lowercase and split on non-word characters
        tokens = re.findall(r'\w+', text.lower())
        # Filter out very short tokens (noise like "a", "is", "an")
        return [t for t in tokens if len(t) > 2]
    
    def build(self, proposition_metadata: list[dict]) -> None:
        """
        Build the BM25 index from proposition metadata.
        
        This is called on app startup, loading propositions from 
        the saved JSON file.
        
        Args:
            proposition_metadata: List of dicts with at least "text" and "id" keys.
                                 Same format as FAISS metadata.
        
        Example:
            >>> store = BM25Store()
            >>> store.build([
            ...     {"id": "p1", "text": "BERT achieves 93.5% on SQuAD"},
            ...     {"id": "p2", "text": "GPT-3 has 175 billion parameters"}
            ... ])
            >>> print(store.count)
            2
        """
        self.documents = proposition_metadata
        
        # Tokenize all documents for BM25
        self._tokenized_docs = [
            self._tokenize(doc.get("text", "")) 
            for doc in self.documents
        ]
        
        if self._tokenized_docs:
            # BM25Okapi takes a list of tokenized documents
            # It builds the term frequency and IDF tables internally
            self.index = BM25Okapi(self._tokenized_docs)
            print(f"📚 BM25 index built with {len(self.documents)} propositions")
        else:
            print("⚠️  BM25 index is empty (no propositions to index)")
    
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search for propositions matching the query keywords.
        
        Args:
            query: The search query text.
            top_k: How many results to return.
        
        Returns:
            List of dicts, each containing:
            - All metadata fields (id, text, doc_id, etc.)
            - "score": the BM25 relevance score (higher = more relevant)
            
            Sorted by score descending (best match first).
        
        Example:
            >>> results = store.search("BERT accuracy SQuAD", top_k=5)
            >>> for r in results:
            ...     print(f"{r['score']:.2f} | {r['text'][:60]}")
        """
        if self.index is None or not self.documents:
            return []
        
        # Tokenize the query the same way we tokenized documents
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores for all documents
        # Returns an array of scores, one per document
        scores = self.index.get_scores(query_tokens)
        
        # Get the indices of the top_k highest scores
        # argsort returns indices sorted ascending; we reverse for descending
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                # BM25 score of 0 means no keyword overlap at all — skip
                continue
            
            result = dict(self.documents[idx])
            result["score"] = score
            results.append(result)
        
        return results
    
    @property
    def count(self) -> int:
        """How many propositions are in the BM25 index."""
        return len(self.documents)


# ============================================================
# SINGLETON
# ============================================================
_bm25_store: BM25Store | None = None


def get_bm25_store() -> BM25Store:
    """Get the singleton BM25 store."""
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
    return _bm25_store
