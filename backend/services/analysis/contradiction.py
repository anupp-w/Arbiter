# ============================================================
# services/analysis/contradiction.py — Contradiction Detection Engine
# ============================================================
#
# THIS IS THE DEMO MOMENT OF THE ENTIRE PROJECT.
#
# WHAT IT DOES:
# -------------
# Takes the top retrieved propositions and checks every pair:
# "Do these two propositions agree, disagree, complement each other,
#  or are they completely unrelated?"
#
# The output is a list of Contradiction objects for any pairs 
# where the relationship is CONTRADICT.
#
# EXAMPLE:
# --------
# Proposition A (from Kaplan et al., 2020):
#   "Model size should be scaled faster than dataset size 
#    when compute budget increases."
#
# Proposition B (from Hoffmann et al., 2022):
#   "Kaplan et al.'s scaling laws are sub-optimal; model size 
#    and dataset size should be scaled in equal proportion."
#
# → Relationship: CONTRADICT ← This triggers a 🔴 Disputed badge!
#
# HOW IT WORKS:
# -------------
# We do pairwise comparison using the LLM. With n=6 propositions,
# that's n*(n-1)/2 = 15 pairs. We send all 15 pairs to Groq in 
# ONE call (not 15 separate calls) to save time and API quota.
#
# PAIRWISE EXPLAINED:
# -------------------
# "Pairwise" means comparing every item against every other item.
# With propositions [A, B, C]:
#   Compare: (A,B), (A,C), (B,C)
# With [A, B, C, D]:
#   Compare: (A,B), (A,C), (A,D), (B,C), (B,D), (C,D)
#
# That's n*(n-1)/2 comparisons. For n=6, that's 15.
# For n=10, that's 45 — too many. We cap at 6 propositions.
# ============================================================

import json
from pathlib import Path
from groq import Groq
from itertools import combinations

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.query import RetrievedProposition, Contradiction
from models.common import PropositionRelationship
from config import settings


# ============================================================
# THE CONTRADICTION DETECTION PROMPT
# ============================================================
# We send ALL pairs to the LLM in a single structured call.
# The LLM returns a JSON array of relationship classifications.
#
# PROMPT DESIGN DECISIONS:
# 1. We show numbered claim pairs (not IDs) for readability
# 2. We define each relationship type with clear criteria
# 3. We demand JSON only — no prose explanation before/after
# 4. We ask for a one-sentence explanation — useful for the UI
# ============================================================

CONTRADICTION_SYSTEM_PROMPT = """You are a scientific claim relationship analyzer.
Given pairs of factual claims from research papers, classify each pair's relationship.

RELATIONSHIP TYPES:
- SUPPORT: Both claims make the same assertion or one confirms the other.
  Example: "BERT achieves 93% on SQuAD" and "BERT outperforms previous models on SQuAD"
  
- CONTRADICT: The claims make incompatible assertions about the same topic.
  Example: "Model size matters most for scaling" vs "Data size matters equally to model size"
  This is a DIRECT factual conflict, not just different perspectives.
  
- COMPLEMENT: Claims cover different aspects of the same topic (no conflict).
  Example: "Transformers use self-attention" and "Transformers require large training data"
  
- UNRELATED: Claims are about different topics entirely.
  Example: "BERT uses WordPiece tokenization" and "LoRA reduces GPU memory usage"

RULES:
- Only classify as CONTRADICT if there is a DIRECT factual conflict.
- Different findings on different datasets or settings = COMPLEMENT, not CONTRADICT.
- Be conservative with CONTRADICT — false positives are worse than false negatives.

Always respond with valid JSON only. No preamble, no markdown.
Output format: {"relationships": [{"pair_id": "0-1", "relationship": "SUPPORT|CONTRADICT|COMPLEMENT|UNRELATED", "explanation": "one sentence"}]}"""


def detect_contradictions(
    propositions: list[RetrievedProposition]
) -> list[Contradiction]:
    """
    Detect contradictions among a set of retrieved propositions.
    
    This function:
    1. Generates all pairs of propositions
    2. Sends them ALL to the LLM in one call
    3. Parses the relationship classifications
    4. Returns Contradiction objects for all CONTRADICT pairs
    
    Args:
        propositions: List of RetrievedProposition objects from retrieval.
                     Should be the top 6 reranked results.
    
    Returns:
        List of Contradiction objects (one per detected contradiction).
        Empty list if no contradictions found.
    
    Example:
        >>> props = [prop_from_kaplan, prop_from_chinchilla, ...]
        >>> contradictions = detect_contradictions(props)
        >>> print(len(contradictions))  # 1 (Kaplan vs Chinchilla)
        >>> print(contradictions[0].relationship)
        # PropositionRelationship.CONTRADICT
    """
    if len(propositions) < 2:
        # Need at least 2 propositions to compare
        return []
    
    # ---- Build all pairs ----
    # combinations([A, B, C, D], 2) gives: (A,B), (A,C), (A,D), (B,C), (B,D), (C,D)
    # We use enumerate to get indices for tracking
    indexed_props = list(enumerate(propositions))
    all_pairs = list(combinations(indexed_props, 2))
    
    # ---- Build the prompt ----
    # Format each pair clearly for the LLM
    pairs_text = ""
    for (idx_a, prop_a), (idx_b, prop_b) in all_pairs:
        pair_id = f"{idx_a}-{idx_b}"
        pairs_text += (
            f"\nPair {pair_id}:\n"
            f"  Claim A (from '{prop_a.doc_title}', {prop_a.section_type} section):\n"
            f"  \"{prop_a.text}\"\n"
            f"  Claim B (from '{prop_b.doc_title}', {prop_b.section_type} section):\n"
            f"  \"{prop_b.text}\"\n"
        )
    
    user_prompt = (
        f"Classify the relationship for each of these {len(all_pairs)} claim pairs:\n"
        f"{pairs_text}"
    )
    
    # ---- Call the LLM ----
    try:
        client = Groq(api_key=settings.groq_api_key)
        
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Deterministic — we want consistent classifications
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        data = json.loads(raw_content)
        relationships = data.get("relationships", [])
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  Contradiction detection error: {e}")
        return []
    
    # ---- Process results ----
    # Build a lookup from pair_id to its relationship
    relationship_map: dict[str, dict] = {
        r["pair_id"]: r for r in relationships
    }
    
    contradictions: list[Contradiction] = []
    
    for (idx_a, prop_a), (idx_b, prop_b) in all_pairs:
        pair_id = f"{idx_a}-{idx_b}"
        result = relationship_map.get(pair_id, {})
        rel_str = result.get("relationship", "UNRELATED").upper()
        explanation = result.get("explanation", "")
        
        # Map string to enum
        try:
            relationship = PropositionRelationship(rel_str.lower())
        except ValueError:
            relationship = PropositionRelationship.UNRELATED
        
        # Only create Contradiction objects for actual contradictions
        if relationship == PropositionRelationship.CONTRADICT:
            contradiction = Contradiction(
                claim_a=prop_a,
                claim_b=prop_b,
                relationship=relationship,
                explanation=explanation
            )
            contradictions.append(contradiction)
            print(f"   🔴 CONTRADICTION detected!")
            print(f"      A ({prop_a.doc_title}): {prop_a.text[:80]}...")
            print(f"      B ({prop_b.doc_title}): {prop_b.text[:80]}...")
    
    print(f"   Contradiction analysis complete: {len(contradictions)} conflicts found "
          f"from {len(all_pairs)} pairs checked")
    
    return contradictions
