# ============================================================
# services/generation/post_processor.py — Hallucination Guard + Confidence
# ============================================================
#
# WHAT THIS FILE DOES:
# --------------------
# After the LLM generates an answer, we need to VERIFY it.
# The LLM is supposed to only cite proposition IDs from our 
# retrieved list. This file checks that every cited ID actually exists.
#
# This is the HALLUCINATION GUARD. Here's why it works:
#
# NORMAL HALLUCINATION DETECTION (bad):
#   "Ask another LLM to judge if the answer looks accurate."
#   Problem: LLMs are bad at judging their own outputs.
#   RAGAS does this — it's LLM-judging-LLM, which is circular.
#
# ARBITER'S HALLUCINATION DETECTION (structural):
#   "Check if every cited source ID actually exists in our database."
#   If claim says source_ids=["prop-abc-123"] but that ID doesn't
#   exist in our retrieved propositions → the LLM invented it.
#   This is a STRUCTURAL check — no LLM judgment needed.
#   It's binary: the source exists or it doesn't.
#
# CONFIDENCE CALIBRATION:
# -----------------------
# We compute a transparent confidence score from 4 factors:
#
#   1. Retrieval quality (avg reranker score) — 0 to 1
#      "Did we find relevant evidence?"
#
#   2. Consensus ratio — 0 to 1
#      "What fraction of claims have multi-source support?"
#
#   3. Source coverage — 0 to 1
#      "What fraction of claims have valid source citations?"
#
#   4. Fallback penalty — 0 or -0.2
#      "Did we have to reformulate the query?"
#
# Final = weighted average of these factors.
# ============================================================

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.query import Claim, RetrievedProposition, ConfidenceBreakdown
from models.common import ClaimStatus


def verify_sources(
    claims: list[Claim],
    retrieved_propositions: list[RetrievedProposition]
) -> list[Claim]:
    """
    Verify that every source ID cited in claims actually exists 
    in the retrieved propositions.
    
    Claims that cite non-existent IDs are flagged with a reduced 
    confidence score and a note in their text.
    
    This is the hallucination guard:
    - Hallucinated source ID → claim is flagged (confidence drops to 0.3)
    - Valid source IDs → claim is kept as-is
    
    Args:
        claims: Generated claims with source_proposition_ids.
        retrieved_propositions: The actual propositions we retrieved.
    
    Returns:
        The same claims, but flagged ones have reduced confidence.
    """
    # Build set of valid proposition IDs
    valid_ids = {p.proposition_id for p in retrieved_propositions}
    
    verified = []
    for claim in claims:
        # Check which source IDs are valid vs hallucinated
        valid_sources = [
            pid for pid in claim.source_proposition_ids 
            if pid in valid_ids
        ]
        hallucinated_sources = [
            pid for pid in claim.source_proposition_ids 
            if pid not in valid_ids
        ]
        
        if hallucinated_sources:
            # The LLM cited a source that doesn't exist.
            # This is a red flag — downgrade confidence significantly.
            print(f"   ⚠️  HALLUCINATION DETECTED: Claim cites non-existent "
                  f"source(s): {hallucinated_sources}")
            print(f"      Claim text: {claim.text[:80]}...")
            
            # Keep only valid sources, reduce confidence
            verified.append(Claim(
                text=claim.text,
                source_proposition_ids=valid_sources,  # Remove hallucinated ones
                status=claim.status,
                confidence=0.3,  # Sharp confidence penalty
                contradiction=claim.contradiction
            ))
        else:
            verified.append(claim)
    
    # Count how many claims had issues
    flagged = sum(1 for c in verified if c.confidence <= 0.3)
    if flagged > 0:
        print(f"   ⚠️  {flagged}/{len(verified)} claims flagged as potentially hallucinated")
    
    return verified


def compute_confidence(
    claims: list[Claim],
    retrieved_propositions: list[RetrievedProposition],
    fallback_triggered: bool = False
) -> ConfidenceBreakdown:
    """
    Compute a transparent, calibrated confidence score for the answer.
    
    Returns a ConfidenceBreakdown with:
    - overall: The final combined score (0 to 1)
    - retrieval_quality: How relevant the retrieved evidence was
    - consensus_ratio: What fraction of claims have multi-source agreement
    - source_coverage: What fraction of claims have valid sources
    - fallback_triggered: Whether query reformulation was needed
    
    Args:
        claims: All classified claims (with status assigned).
        retrieved_propositions: Top retrieved propositions.
        fallback_triggered: Whether the retrieval fallback was used.
    
    Returns:
        ConfidenceBreakdown object with all four components.
    """
    # ---- Factor 1: Retrieval quality ----
    # Average reranker score of retrieved propositions.
    # High reranker scores = we found very relevant evidence.
    if retrieved_propositions:
        avg_reranker = sum(
            p.reranker_score for p in retrieved_propositions
        ) / len(retrieved_propositions)
    else:
        avg_reranker = 0.0
    
    # ---- Factor 2: Consensus ratio ----
    # What fraction of claims are CONSENSUS (multi-source agreement)?
    # More consensus = higher confidence in the answer.
    if claims:
        consensus_count = sum(
            1 for c in claims 
            if c.status == ClaimStatus.CONSENSUS
        )
        consensus_ratio = consensus_count / len(claims)
    else:
        consensus_ratio = 0.0
    
    # ---- Factor 3: Source coverage ----
    # What fraction of claims have at least one valid source?
    # Claims with no valid sources might be hallucinations.
    if claims:
        grounded_count = sum(
            1 for c in claims 
            if len(c.source_proposition_ids) > 0 and c.confidence > 0.3
        )
        source_coverage = grounded_count / len(claims)
    else:
        source_coverage = 0.0
    
    # ---- Compute overall ----
    # Weighted combination:
    #   40% retrieval quality (did we find good evidence?)
    #   30% consensus ratio (do multiple papers agree?)
    #   30% source coverage (are claims grounded?)
    overall = (
        0.4 * avg_reranker +
        0.3 * consensus_ratio +
        0.3 * source_coverage
    )
    
    # Penalty if the retrieval fallback was triggered
    # (means we had to reformulate the query — sign of poor initial retrieval)
    if fallback_triggered:
        overall = max(0.0, overall - 0.2)
    
    # Cap at 0.95 — we should never claim 100% confidence
    overall = min(0.95, overall)
    
    return ConfidenceBreakdown(
        overall=round(overall, 3),
        retrieval_quality=round(avg_reranker, 3),
        consensus_ratio=round(consensus_ratio, 3),
        source_coverage=round(source_coverage, 3),
        fallback_triggered=fallback_triggered
    )
