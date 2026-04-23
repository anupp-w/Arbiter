# ============================================================
# models/document.py — Document, ContextChunk, Proposition
# ============================================================
#
# WHAT ARE THESE MODELS?
# ----------------------
# These are the "blueprints" for the three core objects in our
# ingestion pipeline. When a user uploads a PDF, we create:
#
#   1. Document    → Metadata about the paper (title, authors, etc.)
#   2. ContextChunk → A ~400-token piece of the paper's text
#   3. Proposition  → A single atomic fact extracted from a chunk
#
# REAL-WORLD ANALOGY:
# -------------------
# Imagine you're reading a textbook:
#   - Document      = the entire textbook (title, author, ISBN)
#   - ContextChunk  = one page of the textbook
#   - Proposition   = one specific fact from that page
#     Like: "Water boils at 100°C at sea level."
#     NOT: "This chapter discusses water properties" (too vague)
#     NOT: "Water boils at 100°C and freezes at 0°C" (two facts in one)
#
# WHY PYDANTIC MODELS?
# --------------------
# A Pydantic model is like a TypeScript interface — it declares:
# "this object MUST have these fields with these types."
# If you try to create a Document without a title, Pydantic raises
# an error immediately instead of letting it silently break later.
#
# Example:
#   doc = Document(title="BERT Paper")  → ✅ works (id auto-generated)
#   doc = Document()                     → ❌ error: title is required
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

# Import our custom enums (the "dropdown menus" of allowed values)
from .common import DocumentStatus, SectionType


class Document(BaseModel):
    """
    Represents one research paper in our system.
    
    This is the TOP-LEVEL object. Everything else (chunks, propositions)
    belongs to a Document via the doc_id foreign key.
    
    Think of this like a row in a "papers" database table:
    ┌──────────┬─────────────┬─────────┬────────────┐
    │ id       │ title       │ status  │ date_added │
    ├──────────┼─────────────┼─────────┼────────────┤
    │ abc-123  │ BERT Paper  │ done    │ 2024-01-15 │
    │ def-456  │ GPT-3 Paper │ working │ 2024-01-16 │
    └──────────┴─────────────┴─────────┴────────────┘
    """
    
    # ---- Identity ----
    # uuid4() generates a random unique ID like "a3f2b8c1-..."
    # We use strings instead of UUID objects because JSON doesn't 
    # understand UUID objects, but it understands strings.
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this document."
    )
    
    # ---- Core Metadata ----
    title: str = Field(
        ...,  # The ... means "REQUIRED — no default value"
        description="Title of the paper."
    )
    authors: list[str] = Field(
        default_factory=list,
        description="List of author names. Empty if we couldn't extract them."
    )
    source_filename: str = Field(
        default="",
        description="Original filename of the uploaded PDF."
    )
    
    # ---- Processing State ----
    # This tracks where the document is in the ingestion pipeline.
    # The UI polls this to show a progress indicator.
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING,
        description="Current ingestion status."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="If status is FAILED, what went wrong."
    )
    
    # ---- Extracted Info ----
    # These get filled in during ingestion (Phase 1).
    num_chunks: int = Field(
        default=0,
        description="How many context chunks were created from this document."
    )
    num_propositions: int = Field(
        default=0,
        description="How many atomic propositions were extracted."
    )
    summary: Optional[str] = Field(
        default=None,
        description="One-paragraph LLM-generated summary of the paper."
    )
    domain: Optional[str] = Field(
        default=None,
        description="Research domain: ML, NLP, CV, etc."
    )
    
    # ---- Timestamps ----
    date_ingested: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this document was added to the system."
    )
    date_completed: Optional[datetime] = Field(
        default=None,
        description="When ingestion finished (None if still processing)."
    )


class ContextChunk(BaseModel):
    """
    A ~400-token piece of text from a document, aligned to section boundaries.
    
    WHY CHUNKS?
    -----------
    LLMs have a limited context window. We can't feed an entire 30-page 
    paper into one prompt. So we split it into digestible pieces.
    
    WHY SECTION-AWARE?
    ------------------
    Naive chunking (split every N tokens) can cut a sentence in half:
      "The model achieves 94.2% accuracy on | MMLU, surpassing the baseline."
    
    Section-aware chunking respects boundaries:
      Chunk 1 (Results): "The model achieves 94.2% accuracy on MMLU, surpassing..."
      Chunk 2 (Conclusion): "In conclusion, we demonstrated that..."
    
    This preserves meaning AND gives us metadata about WHERE in the paper 
    each chunk came from (Results > Introduction for authority).
    
    ANALOGY:
    --------
    If the Document is a book, a ContextChunk is one page.
    The page knows which chapter it belongs to (section_type)
    and where it appears (position: "early", "middle", "late").
    """
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this chunk."
    )
    doc_id: str = Field(
        ...,
        description="Which document this chunk came from (foreign key)."
    )
    text: str = Field(
        ...,
        description="The actual text content of this chunk."
    )
    section_type: SectionType = Field(
        default=SectionType.OTHER,
        description="Which section of the paper this chunk belongs to."
    )
    token_count: int = Field(
        default=0,
        description="Number of tokens in this chunk (counted by tiktoken)."
    )
    position: str = Field(
        default="middle",
        description="Where in the document: 'early', 'middle', or 'late'."
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Which PDF page(s) this chunk spans."
    )


class Proposition(BaseModel):
    """
    A single atomic factual claim extracted from a context chunk.
    
    THIS IS THE CORE DIFFERENTIATOR OF ARBITER.
    
    WHAT IS A PROPOSITION?
    ----------------------
    An atomic, self-contained factual claim. One fact per proposition.
    It must make sense even if you've never read the paper.
    
    GOOD propositions (what we want):
      ✅ "BERT achieves 93.5% accuracy on the SQuAD 2.0 benchmark."
      ✅ "The Transformer architecture uses multi-head self-attention 
          instead of recurrence."
      ✅ "Pre-training on 16GB of text data improves downstream task 
          performance by an average of 12% compared to training from scratch."
    
    BAD propositions (what we filter out):
      ❌ "The results are shown in Table 3." (not a factual claim)
      ❌ "It improved." (what improved? by how much? compared to what?)
      ❌ "BERT is good and also fast and outperforms everything." 
         (multiple claims crammed into one)
    
    WHY THIS MATTERS:
    -----------------
    When we search for "how accurate is BERT on SQuAD?", a proposition 
    like "BERT achieves 93.5% accuracy on SQuAD 2.0" is a MUCH better 
    search result than a 400-token chunk that mentions SQuAD somewhere 
    in the middle. Proposition indexing makes retrieval dramatically 
    more precise.
    
    This technique comes from the RAPTOR and Dense Passage Retrieval 
    research papers. Nobody does this in tutorials — mentioning it in 
    an interview immediately signals you've read actual research.
    """
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this proposition."
    )
    doc_id: str = Field(
        ...,
        description="Which document this proposition was extracted from."
    )
    chunk_id: str = Field(
        ...,
        description="Which context chunk this proposition was extracted from."
    )
    text: str = Field(
        ...,
        description="The actual proposition text — one atomic factual claim."
    )
    section_type: SectionType = Field(
        default=SectionType.OTHER,
        description="Which section the source chunk belongs to."
    )
    extraction_confidence: float = Field(
        default=1.0,
        description=(
            "How confident we are in this extraction (0.0 to 1.0). "
            "1.0 = clearly a factual claim. "
            "Lower values = might be opinion, might be vague."
        )
    )
    # We store the document title directly on each proposition
    # so we can display "Source: BERT Paper" without a database join.
    # This is intentional denormalization — a small price in storage
    # for a huge gain in simplicity.
    doc_title: str = Field(
        default="",
        description="Title of the source document (denormalized for display)."
    )
