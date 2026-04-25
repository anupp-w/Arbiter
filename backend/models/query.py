# ============================================================
# models/query.py - Query Request, Result, Claims, Contradictions
# ============================================================
#
# WHAT ARE THESE MODELS?
# ----------------------
# These define the "shape" of everything that happens when a user 
# asks a question. The flow is:
#
#   User types a question
#       ↓
#   QueryRequest (what they asked)
#       ↓
#   [Retrieval → Reranking → Contradiction Detection → Generation]
#       ↓
#   QueryResult (the structured answer with claims and badges)
#
# ANALOGY:
# --------
# QueryRequest  = the question on a test paper
# QueryResult   = your answer sheet, with:
#   - The answer itself
#   - Which textbook pages you cited
#   - Whether different textbooks agreed or disagreed
#   - How confident you are in your answer
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid

from .common import ClaimStatus, PropositionRelationship


class QueryRequest(BaseModel):
    """
    What the user sends when they ask a question.
    
    Simple on purpose - the complexity is in the RESPONSE, not the request.
    """
    query_text: str = Field(
        ...,
        description="The user's question. Example: 'How do scaling laws differ between papers?'"
    )


class RetrievedProposition(BaseModel):
    """
    A proposition that was retrieved as relevant to the user's query.
    
    This is a Proposition (from document.py) PLUS retrieval-specific 
    metadata like the relevance score.
    
    Think of it like a search result on Google:
    - The proposition text = the search result title
    - The score = how relevant Google thinks it is
    - The doc_title = the website name
    - The section_type = which page of the website
    """
    proposition_id: str = Field(..., description="ID of the original proposition.")
    text: str = Field(..., description="The proposition text.")
    doc_id: str = Field(..., description="Which document it came from.")
    doc_title: str = Field(..., description="Title of the source document.")
    chunk_id: str = Field(..., description="Which chunk it was extracted from.")
    section_type: str = Field(default="other", description="Section of the paper.")
    
    # ---- Retrieval Scores ----
    # These show HOW this proposition was found and how relevant it is.
    # A high reranker_score is the strongest signal of relevance.
    dense_score: float = Field(
        default=0.0,
        description="Cosine similarity from FAISS dense retrieval (0 to 1)."
    )
    sparse_score: float = Field(
        default=0.0,
        description="BM25 keyword matching score."
    )
    rrf_score: float = Field(
        default=0.0,
        description="Combined RRF fusion score from dense + sparse."
    )
    reranker_score: float = Field(
        default=0.0,
        description=(
            "Cross-encoder reranker score (0 to 1). "
            "This is the most accurate relevance signal. "
            "A score > 0.7 means 'definitely relevant'. "
            "A score < 0.3 means 'probably not relevant'."
        )
    )


class Contradiction(BaseModel):
    """
    A detected disagreement between two propositions.
    
    THIS IS THE MONEY SHOT OF THE ENTIRE PROJECT.
    
    When we find that Paper A says "X is true" and Paper B says 
    "X is false", we create a Contradiction object that captures both 
    sides. The UI shows this as a red "DISPUTED" badge that expands 
    to show both sides with their sources.
    
    Example:
    ┌─────────────────────────────────────────────────┐
    │  DISPUTED                                      │
    │                                                   │
    │ Claim A (Kaplan et al., 2020):                   │
    │ "Model size should be scaled faster than          │
    │  dataset size for optimal compute allocation."    │
    │                                                   │
    │ Claim B (Hoffmann et al., 2022):                 │
    │ "Model size and dataset size should be scaled     │
    │  equally - Kaplan's scaling laws were suboptimal."│
    │                                                   │
    │ Why they disagree: Different experimental setups  │
    │ and compute budgets led to opposite conclusions   │
    │ about optimal scaling ratios.                     │
    └─────────────────────────────────────────────────┘
    """
    
    claim_a: RetrievedProposition = Field(
        ..., description="The first proposition in the disagreement."
    )
    claim_b: RetrievedProposition = Field(
        ..., description="The second (conflicting) proposition."
    )
    relationship: PropositionRelationship = Field(
        ..., description="How these two propositions relate to each other."
    )
    explanation: str = Field(
        default="",
        description="One-sentence explanation of WHY they disagree."
    )


class Claim(BaseModel):
    """
    A single claim in our generated answer, with its source and status.
    
    This is what makes Arbiter's output fundamentally different from 
    normal RAG. Instead of a blob of text, every sentence is tagged:
    
    Normal RAG output:
      "Transformers outperform RNNs. Larger models are better. 
       Scaling data equally with model size is important."
      (Which sources? Any disagreements? Who knows!)
    
    Arbiter output:
       "Transformers outperform RNNs." [BERT, GPT-3, RoBERTa]
       "Larger models are better." [Kaplan]  [Chinchilla disagrees]
       "RAG reduces hallucination." [Lewis et al. only]
    """
    
    text: str = Field(
        ..., description="The claim text - one sentence."
    )
    source_proposition_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of the propositions that support this claim. "
            "If this list is empty after generation, it means the LLM "
            "made this claim up (hallucination) and we flag it with a warning."
        )
    )
    status: ClaimStatus = Field(
        ..., description="Consensus, Disputed, Single-Source, or Insufficient."
    )
    confidence: float = Field(
        default=1.0,
        description="How confident we are in this specific claim (0.0 to 1.0)."
    )
    # If this claim is disputed, which contradiction object describes the conflict?
    contradiction: Optional[Contradiction] = Field(
        default=None,
        description="If status is DISPUTED, the details of the contradiction."
    )


class ConfidenceBreakdown(BaseModel):
    """
    A transparent breakdown of WHY we're confident (or not) in our answer.
    
    Instead of just saying "confidence: 0.73" (which means nothing), 
    we show WHAT went into that number:
    
    Example:
    ┌────────────────────────────────────┐
    │ Overall Confidence: 73%            │
    │                                    │
    │ Retrieval Quality:     0.85      │
    │ Consensus Ratio:       0.60      │
    │ Source Coverage:       0.75      │
    │ No Fallback Triggered: Yes       │
    └────────────────────────────────────┘
    
    This transparency is what senior engineers look for.
    An opaque "0.73" is useless. A breakdown is actionable.
    """
    
    overall: float = Field(
        ..., description="Final confidence score (0.0 to 1.0)."
    )
    retrieval_quality: float = Field(
        default=0.0,
        description=(
            "Average reranker score of top results. "
            "High = we found very relevant evidence. "
            "Low = the evidence might not match the query well."
        )
    )
    consensus_ratio: float = Field(
        default=0.0,
        description=(
            "Proportion of claims that are CONSENSUS. "
            "1.0 = all claims have multi-source agreement. "
            "0.0 = no agreement found (concerning)."
        )
    )
    source_coverage: float = Field(
        default=0.0,
        description=(
            "Proportion of claims with at least one valid source ID. "
            "1.0 = every claim is grounded. "
            "< 1.0 = some claims might be hallucinated."
        )
    )
    fallback_triggered: bool = Field(
        default=False,
        description="Whether the retrieval fallback (query reformulation) was triggered."
    )


class QueryResult(BaseModel):
    """
    The COMPLETE response to a user's query. This is what the API returns 
    and what the Streamlit UI renders.
    
    It contains EVERYTHING:
    - The prose answer (what the user reads)
    - Individual claims with badges (the visual wow factor)
    - Contradictions (the demo moment)
    - Confidence breakdown (the transparency signal)
    - Retrieved sources (for citation)
    - Trace ID (for observability/debugging)
    """
    
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this query result (used for trace linking)."
    )
    query_text: str = Field(
        ..., description="The original question that was asked."
    )
    
    # ---- The Answer ----
    main_answer: str = Field(
        default="",
        description=(
            "The prose answer - a readable paragraph that synthesizes "
            "the evidence. This is what gets displayed at the top of "
            "the results panel."
        )
    )
    
    # ---- Structured Claims ----
    # This is where the magic is visible. Each claim has a colored badge.
    claims: list[Claim] = Field(
        default_factory=list,
        description="Individual claims extracted from the answer, each with a status badge."
    )
    
    # ---- Contradictions Found ----
    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="All detected contradictions between sources."
    )
    
    # ---- Sources Used ----
    retrieved_propositions: list[RetrievedProposition] = Field(
        default_factory=list,
        description="All propositions that were retrieved and used to generate the answer."
    )
    
    # ---- Confidence ----
    confidence: ConfidenceBreakdown = Field(
        default_factory=lambda: ConfidenceBreakdown(overall=0.0),
        description="Transparent confidence breakdown."
    )
    
    # ---- Metadata ----
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this query was processed."
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Phoenix trace ID for observability. Links to the trace UI."
    )
    processing_time_seconds: float = Field(
        default=0.0,
        description="Total time to process this query (end-to-end)."
    )
