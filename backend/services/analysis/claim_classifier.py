# ============================================================
# services/analysis/claim_classifier.py — Consensus/Disputed/Single-Source
# ============================================================
#
# WHAT THIS FILE DOES:
# --------------------
# After we generate an answer with individual claims, we need to 
# assign each claim a STATUS:
#
#   🟢 CONSENSUS    → Multiple independent papers agree
#   🔴 DISPUTED     → Papers directly contradict each other
#   🟡 SINGLE_SOURCE → Only one paper mentions this
#   ⚪ INSUFFICIENT  → Not enough evidence found
#
# HOW WE DETERMINE STATUS:
# -------------------------
# For each claim in the generated answer:
#   1. Look at which proposition IDs it cites (source_ids)
#   2. Look at which documents those propositions came from
#   3. If source_ids are from ≥2 DIFFERENT documents → CONSENSUS
#   4. If any detected contradiction involves its sources → DISPUTED
#   5. If source_ids are all from ONE document → SINGLE_SOURCE
#   6. If no source_ids → mark as unverified (hallucination warning)
# ============================================================

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.query import Claim, RetrievedProposition, Contradiction
from models.common import ClaimStatus


def classify_claim(
    claim_text: str,
    source_prop_ids: list[str],
    retrieved_propositions: list[RetrievedProposition],
    contradictions: list[Contradiction]
) -> ClaimStatus:
    """
    Determine the consensus status of a claim based on its sources.
    
    Args:
        claim_text: The text of the claim (for logging).
        source_prop_ids: IDs of propositions that support this claim.
        retrieved_propositions: All propositions retrieved for this query.
        contradictions: All contradictions detected in this query.
    
    Returns:
        ClaimStatus: CONSENSUS, DISPUTED, SINGLE_SOURCE, or INSUFFICIENT.
    """
    # Build a lookup: proposition_id → proposition object
    prop_lookup: dict[str, RetrievedProposition] = {
        p.proposition_id: p for p in retrieved_propositions
    }
    
    # ---- Case 1: No sources ---- 
    if not source_prop_ids:
        return ClaimStatus.INSUFFICIENT
    
    # ---- Get the source propositions ----
    source_props = [
        prop_lookup[pid] 
        for pid in source_prop_ids 
        if pid in prop_lookup
    ]
    
    if not source_props:
        return ClaimStatus.INSUFFICIENT
    
    # ---- Check for contradictions first ----
    # If any of the source propositions are involved in a detected contradiction,
    # this claim is DISPUTED — even if it comes from multiple sources.
    # Being disputed overrides being consensus.
    source_prop_id_set = set(source_prop_ids)
    
    for contradiction in contradictions:
        if (contradiction.claim_a.proposition_id in source_prop_id_set or
                contradiction.claim_b.proposition_id in source_prop_id_set):
            return ClaimStatus.DISPUTED
    
    # ---- Check how many unique documents support this claim ----
    unique_doc_ids = {p.doc_id for p in source_props}
    
    if len(unique_doc_ids) >= 2:
        # Multiple independent documents agree → CONSENSUS
        return ClaimStatus.CONSENSUS
    elif len(unique_doc_ids) == 1:
        # Only one document mentions this → SINGLE_SOURCE
        return ClaimStatus.SINGLE_SOURCE
    else:
        return ClaimStatus.INSUFFICIENT


def classify_all_claims(
    claims: list[Claim],
    retrieved_propositions: list[RetrievedProposition],
    contradictions: list[Contradiction]
) -> list[Claim]:
    """
    Classify the status of ALL claims in the generated answer.
    
    Also attaches the relevant Contradiction object to any DISPUTED claim
    so the UI can show both sides.
    
    Args:
        claims: Claims from the generated answer (with source_proposition_ids).
        retrieved_propositions: All propositions retrieved for this query.
        contradictions: All contradictions detected.
    
    Returns:
        The same claims list, but with status and contradiction fields filled in.
    """
    # Build contradiction lookup by proposition ID for quick lookup
    # key: proposition_id, value: Contradiction object
    contradiction_by_prop: dict[str, Contradiction] = {}
    for c in contradictions:
        contradiction_by_prop[c.claim_a.proposition_id] = c
        contradiction_by_prop[c.claim_b.proposition_id] = c
    
    classified = []
    for claim in claims:
        # Determine the status
        status = classify_claim(
            claim_text=claim.text,
            source_prop_ids=claim.source_proposition_ids,
            retrieved_propositions=retrieved_propositions,
            contradictions=contradictions
        )
        
        # For DISPUTED claims, attach the contradiction details
        relevant_contradiction = None
        if status == ClaimStatus.DISPUTED:
            for prop_id in claim.source_proposition_ids:
                if prop_id in contradiction_by_prop:
                    relevant_contradiction = contradiction_by_prop[prop_id]
                    break
        
        classified.append(Claim(
            text=claim.text,
            source_proposition_ids=claim.source_proposition_ids,
            status=status,
            confidence=claim.confidence,
            contradiction=relevant_contradiction
        ))
    
    # Log summary
    status_counts = {}
    for c in classified:
        status_counts[c.status.value] = status_counts.get(c.status.value, 0) + 1
    print(f"   Claim statuses: {status_counts}")
    
    return classified
