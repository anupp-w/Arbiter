# ============================================================
# services/retrieval/embedder.py - Text → Vector Embedding
# ============================================================
#
# WHAT IS AN EMBEDDING?
# ---------------------
# Imagine you could represent any sentence as a POINT in space.
# Sentences with similar meanings would be CLOSE together,
# and unrelated sentences would be FAR apart.
#
# That's exactly what an embedding model does:
#   "The cat sat on the mat"  →  [0.12, -0.45, 0.78, ... (384 numbers)]
#   "A kitten rested on a rug" →  [0.11, -0.44, 0.77, ... (384 numbers)]
#   "Stock prices rose today"  →  [0.89, 0.23, -0.56, ... (384 numbers)]
#
# The first two vectors are CLOSE (similar meaning).
# The third is FAR away (different topic entirely).
#
# We use this to search: embed the user's question, then find 
# which stored propositions have the closest vectors.
#
# THE MODEL WE USE:
# -----------------
# BAAI/bge-small-en-v1.5
# - "bge" = Beijing Academy of AI General Embedding
# - "small" = 33M parameters (runs fast on CPU, even faster on GPU)
# - "en" = English only
# - "v1.5" = latest version
# - Produces 384-dimensional vectors
# - Near the TOP of the MTEB benchmark (the "leaderboard" for embeddings)
# - Completely free, runs locally, no API key needed
#
# GPU vs CPU:
# -----------
# Your GTX 1650Ti has 4GB VRAM. bge-small uses ~130MB.
# So it fits easily on GPU → embeddings compute ~10x faster.
# The code auto-detects GPU and uses it if available.
# ============================================================

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import Union

# We import settings to know which model to load.
# This is the "one source of truth" pattern from config.py.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings


class Embedder:
    """
    Wraps the sentence-transformers library to provide a simple interface:
        embedder = Embedder()
        vector = embedder.embed("some text")         # one text
        vectors = embedder.embed_batch(["a", "b"])    # multiple texts
    
    The model is loaded ONCE when you create the Embedder object,
    then reused for all subsequent embed() calls. Loading is the 
    slow part (~2-5 seconds); embedding after that is fast (~10ms).
    """
    
    def __init__(self):
        """
        Load the embedding model into memory (GPU if available, else CPU).
        
        This is called ONCE when the app starts. We use Streamlit's 
        @st.cache_resource decorator (in the frontend) to ensure we 
        don't accidentally load it multiple times.
        """
        # ---- Device Selection ----
        # CUDA = NVIDIA GPU. If you have one, use it. If not, CPU works fine.
        # "cuda" = GPU, "cpu" = CPU
        # torch.cuda.is_available() returns True if NVIDIA drivers are installed
        # and a compatible GPU is detected.
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f" Embedder using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("  Embedder using CPU (slower but works fine)")
        
        # ---- Load the Model ----
        # SentenceTransformer downloads the model from HuggingFace on first run
        # (~90MB download), then caches it locally (~/.cache/huggingface/).
        # Subsequent runs load from cache instantly.
        print(f" Loading embedding model: {settings.embedding_model}...")
        self.model = SentenceTransformer(
            settings.embedding_model,
            device=self.device
        )
        print(f" Embedding model loaded! Dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Sanity check: make sure the model's dimension matches our config.
        # If someone changes the model in .env but forgets to update the dimension,
        # this will catch it immediately instead of causing mysterious bugs later.
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != settings.embedding_dimension:
            raise ValueError(
                f"Model dimension ({actual_dim}) doesn't match config "
                f"({settings.embedding_dimension}). Update EMBEDDING_DIMENSION in .env."
            )
    
    def embed(self, text: str) -> np.ndarray:
        """
        Convert a single text string into a vector (numpy array).
        
        Args:
            text: Any string, e.g., "BERT achieves 93.5% on SQuAD."
        
        Returns:
            A numpy array of shape (384,) - 384 floating-point numbers 
            that represent the meaning of the text.
        
        Example:
            >>> embedder = Embedder()
            >>> vec = embedder.embed("Transformers use self-attention.")
            >>> print(vec.shape)  # (384,)
            >>> print(vec[:5])    # [-0.023, 0.156, -0.089, ...]
        """
        # encode() is the core method from sentence-transformers.
        # It handles tokenization, padding, and the neural network forward pass.
        # normalize_embeddings=True makes cosine similarity work correctly.
        #   Without normalization: cosine_sim = dot(a,b) / (|a| * |b|)
        #   With normalization: cosine_sim = dot(a,b) - simpler and faster!
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(embedding, dtype=np.float32)
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert multiple texts into vectors at once (much faster than one-by-one).
        
        WHY BATCH?
        ----------
        GPUs are like buses - they're most efficient when full.
        Embedding 1 text at a time = sending a bus with 1 passenger.
        Embedding 32 at a time = full bus, same fuel cost.
        
        On your 1650Ti (4GB VRAM), batch_size=32 is safe.
        Each text becomes ~384 floats × 4 bytes = ~1.5KB per embedding.
        32 texts = ~48KB - trivial for 4GB VRAM.
        
        Args:
            texts: List of strings to embed.
            batch_size: How many to process at once. 32 is a safe default
                       for your 4GB GPU. Increase if you have more VRAM.
        
        Returns:
            numpy array of shape (len(texts), 384) - one row per text.
        
        Example:
            >>> vecs = embedder.embed_batch(["hello", "world", "test"])
            >>> print(vecs.shape)  # (3, 384)
        """
        if not texts:
            # Return an empty array with the right shape if no texts given.
            # This prevents crashes when processing empty documents.
            return np.array([], dtype=np.float32).reshape(0, settings.embedding_dimension)
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,  # Only show progress for large batches
            batch_size=batch_size
        )
        return np.array(embeddings, dtype=np.float32)


# ============================================================
# MODULE-LEVEL SINGLETON
# ============================================================
# We create ONE Embedder instance here. Every file that needs 
# embeddings imports this same object:
#
#   from services.retrieval.embedder import embedder_instance
#   vec = embedder_instance.embed("some text")
#
# This prevents loading the model multiple times (which would 
# waste GPU memory and take 5+ seconds each time).
#
# NOTE: We use lazy initialization (created on first import)
# because the model download might fail if there's no internet.
# ============================================================
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """
    Get the singleton Embedder instance.
    Creates it on first call, returns the same one after that.
    
    This pattern is called "lazy initialization" - we don't load 
    the model until someone actually needs it. This makes imports 
    fast and prevents crashes if the model isn't downloaded yet.
    """
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
