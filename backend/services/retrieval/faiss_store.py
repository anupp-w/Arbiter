# ============================================================
# services/retrieval/faiss_store.py - FAISS Vector Index Manager
# ============================================================
#
# WHAT IS FAISS?
# --------------
# FAISS (Facebook AI Similarity Search) is a library by Meta that 
# stores vectors and finds the most similar ones FAST.
#
# Think of it like a library catalog, but instead of searching by 
# title or author, you search by MEANING. You give it a question 
# vector, and it finds the stored vectors closest to it in meaning.
#
# ANALOGY:
# --------
# Imagine a huge room full of books, and each book has a GPS coordinate 
# based on its topic. FAISS is like saying "find the 10 books closest 
# to THIS GPS point" - but in 384-dimensional space instead of 2D.
#
# WHY TWO INDEXES?
# ----------------
# We maintain TWO separate FAISS indexes:
#
# 1. PROPOSITION INDEX - stores embeddings of atomic facts
#    "BERT achieves 93.5% on SQuAD" → vector → stored here
#    GOOD FOR: finding precise factual answers
#
# 2. CHUNK INDEX - stores embeddings of ~400-token chunks
#    "In this section we describe the BERT model..." → vector → stored here
#    GOOD FOR: getting full context around a finding
#
# When a user asks a question, we search BOTH indexes, then combine results.
# This dual-index approach is called "proposition + context retrieval" and 
# it's what makes Arbiter's retrieval better than vanilla RAG.
#
# FAISS vs PINECONE:
# ------------------
# Original spec used Pinecone (cloud). We use FAISS (local) because:
# - Free, no API key needed
# - No internet required
# - Fast enough for demo scale (under 100k vectors)
# - Data stays on your machine
# The trade-off: FAISS doesn't have built-in metadata filtering.
# We handle that in Python after retrieving candidates.
# ============================================================

import faiss
import numpy as np
import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings


class FAISSStore:
    """
    Manages a single FAISS index with associated metadata.
    
    FAISS only stores vectors (lists of numbers). It doesn't store 
    text, IDs, or any other metadata. So we maintain a SEPARATE 
    metadata list that maps each vector's position to its metadata.
    
    It's like a phone book where:
    - FAISS stores the GPS coordinates of each house
    - Our metadata list stores the names and addresses at each position
    
    Position 0 in FAISS corresponds to position 0 in metadata, etc.
    """
    
    def __init__(self, name: str, dimension: int = None):
        """
        Create a new FAISS index store.
        
        Args:
            name: A name for this index (e.g., "propositions" or "chunks").
                  Used for file naming when saving/loading.
            dimension: Vector dimension (384 for bge-small). 
                       If None, uses the setting from config.
        """
        self.name = name
        self.dimension = dimension or settings.embedding_dimension
        
        # ---- Create the FAISS index ----
        # IndexFlatIP = "Flat Index with Inner Product similarity"
        #
        # "Flat" = brute-force search (checks every vector).
        # For <100k vectors, this is fast enough (~1ms).
        # For millions of vectors, you'd use IndexIVFFlat (approximate search).
        #
        # "IP" = Inner Product. Since we normalize our embeddings,
        # Inner Product = Cosine Similarity. Higher = more similar.
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # ---- Metadata storage ----
        # Each entry maps to the vector at the same position in the FAISS index.
        # Example metadata entry:
        # {
        #     "id": "prop-abc-123",
        #     "doc_id": "doc-xyz",
        #     "text": "BERT achieves 93.5% on SQuAD.",
        #     "section_type": "results",
        #     "doc_title": "BERT Paper"
        # }
        self.metadata: list[dict] = []
        
        # Where to save/load the index and metadata on disk
        self.index_path = settings.faiss_dir / f"{name}.index"
        self.metadata_path = settings.faiss_dir / f"{name}_metadata.json"
    
    def add(self, vectors: np.ndarray, metadata_list: list[dict]) -> None:
        """
        Add vectors and their metadata to the index.
        
        Think of this as "registering new books in the library."
        Each vector gets stored in FAISS, and its metadata gets 
        appended to our metadata list.
        
        Args:
            vectors: numpy array of shape (n, 384) - n embedding vectors.
            metadata_list: List of n dicts, one per vector, containing 
                          the text and identifiers for each vector.
        
        Example:
            >>> store.add(
            ...     vectors=np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]]),
            ...     metadata_list=[
            ...         {"id": "p1", "text": "BERT is a transformer model."},
            ...         {"id": "p2", "text": "GPT-3 has 175B parameters."}
            ...     ]
            ... )
        """
        if len(vectors) == 0:
            return
            
        # Sanity checks
        if len(vectors) != len(metadata_list):
            raise ValueError(
                f"Mismatch: {len(vectors)} vectors but {len(metadata_list)} metadata entries. "
                "Each vector must have exactly one metadata entry."
            )
        
        if vectors.ndim == 1:
            # Single vector - reshape to (1, dimension)
            vectors = vectors.reshape(1, -1)
        
        # Ensure float32 - FAISS requires this specific type
        vectors = vectors.astype(np.float32)
        
        # Add to FAISS index
        self.index.add(vectors)
        
        # Add to metadata list (in the same order!)
        self.metadata.extend(metadata_list)
        
        print(f"    Added {len(vectors)} vectors to '{self.name}' index "
              f"(total: {self.index.ntotal})")
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Find the top_k most similar vectors to the query.
        
        This is the core search operation. FAISS computes the similarity 
        between the query vector and ALL stored vectors, then returns 
        the top_k most similar ones.
        
        Args:
            query_vector: The embedded query, shape (384,) or (1, 384).
            top_k: How many results to return.
        
        Returns:
            List of dicts, each containing:
            - All metadata fields (id, text, doc_id, etc.)
            - "score": the similarity score (higher = more similar)
            
            Sorted by score descending (best match first).
        
        Example:
            >>> results = store.search(query_vec, top_k=5)
            >>> for r in results:
            ...     print(f"{r['score']:.3f} | {r['text'][:60]}")
            0.847 | BERT achieves 93.5% F1 score on the SQuAD 2.0 benchmark.
            0.812 | The Transformer model uses multi-head self-attention...
            0.798 | Pre-training improves downstream performance by 4.5%...
        """
        if self.index.ntotal == 0:
            # No vectors stored yet - return empty results
            return []
        
        # Reshape if needed: FAISS expects shape (1, dimension) for a single query
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype(np.float32)
        
        # Don't request more results than we have vectors
        actual_k = min(top_k, self.index.ntotal)
        
        # ---- FAISS search ----
        # Returns two arrays:
        # - scores: shape (1, actual_k) - similarity scores
        # - indices: shape (1, actual_k) - positions in the index
        #
        # Example: scores = [[0.847, 0.812, 0.798]], indices = [[42, 17, 3]]
        # This means vector #42 is most similar (score 0.847), etc.
        scores, indices = self.index.search(query_vector, actual_k)
        
        # ---- Build results ----
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                # FAISS returns -1 for "no result found" in some configurations
                continue
            
            # Get the metadata for this vector position
            result = dict(self.metadata[idx])  # Copy the dict
            result["score"] = float(score)  # Add the similarity score
            results.append(result)
        
        return results
    
    def save(self) -> None:
        """
        Save the FAISS index and metadata to disk.
        
        Called after ingesting documents so we don't lose the index 
        when the server restarts. Two files are saved:
        - {name}.index - the FAISS binary index file
        - {name}_metadata.json - the metadata JSON file
        """
        # Create the directory if it doesn't exist
        settings.faiss_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index (binary format)
        faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata (JSON format)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f" Saved '{self.name}' index ({self.index.ntotal} vectors) to disk")
    
    def load(self) -> bool:
        """
        Load a previously saved FAISS index and metadata from disk.
        
        Returns:
            True if loaded successfully, False if files don't exist.
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            print(f"  No saved '{self.name}' index found. Starting fresh.")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            print(f" Loaded '{self.name}' index ({self.index.ntotal} vectors) from disk")
            
            # Sanity check: metadata count should match vector count
            if len(self.metadata) != self.index.ntotal:
                print(f"  Warning: metadata count ({len(self.metadata)}) != "
                      f"vector count ({self.index.ntotal}). Index may be corrupted.")
            
            return True
            
        except Exception as e:
            print(f" Error loading '{self.name}' index: {e}")
            # Reset to empty state
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            return False
    
    def clear(self) -> None:
        """
        Remove all vectors and metadata. Start fresh.
        Useful for testing or rebuilding the index.
        """
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        print(f"  Cleared '{self.name}' index")
    
    @property
    def count(self) -> int:
        """How many vectors are stored in this index."""
        return self.index.ntotal


# ============================================================
# DUAL INDEX SINGLETONS
# ============================================================
# We create TWO index instances - one for propositions, one for chunks.
# These are loaded from disk on startup (if saved data exists).
# ============================================================

_proposition_store: FAISSStore | None = None
_chunk_store: FAISSStore | None = None


def get_proposition_store() -> FAISSStore:
    """Get the singleton FAISS store for propositions."""
    global _proposition_store
    if _proposition_store is None:
        _proposition_store = FAISSStore("propositions")
        _proposition_store.load()  # Try to load from disk
    return _proposition_store


def get_chunk_store() -> FAISSStore:
    """Get the singleton FAISS store for context chunks."""
    global _chunk_store
    if _chunk_store is None:
        _chunk_store = FAISSStore("chunks")
        _chunk_store.load()  # Try to load from disk
    return _chunk_store
