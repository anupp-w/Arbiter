# ============================================================
# services/generation/answer_generator.py - Structured Answer Generation
# ============================================================
#
# WHAT THIS FILE DOES:
# --------------------
# Takes the retrieved + reranked propositions and the contradiction 
# analysis, and asks the LLM to generate a STRUCTURED answer where 
# every claim cites specific proposition IDs.
#
# THIS IS NOT NORMAL RAG GENERATION.
# -----------------------------------
# Normal RAG: "Here are 5 chunks of text, write an answer."
# → The LLM blends everything together, makes up connections,
#   loses track of which paper said what. Hallucination risk is high.
#
# Arbiter: "Here are 6 specific propositions with their IDs.
#           Write an answer where each sentence you make is tagged 
#           with the proposition ID(s) that support it. If you make
#           a claim that isn't in the propositions, don't include it."
# → Every claim is grounded. We can verify it programmatically.
#   This is the hallucination guard we build in post_processor.py.
#
# THE OUTPUT FORMAT:
# ------------------
# The LLM returns a JSON object with this structure:
# {
#   "answer": "Prose answer text...",
#   "claims": [
#     {
#       "text": "BERT achieves 93.5% on SQuAD.",
#       "source_ids": ["prop-abc-123", "prop-def-456"],
#       "confidence": 0.9
#     },
#     ...
#   ]
# }
#
# We then run this through the claim classifier to add 
# CONSENSUS/DISPUTED/SINGLE_SOURCE statuses.
# ============================================================

import json
from pathlib import Path
from groq import Groq

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.query import RetrievedProposition, Claim, QueryResult, ConfidenceBreakdown
from models.common import ClaimStatus
from config import settings


GENERATION_SYSTEM_PROMPT = """You are a precise research synthesis assistant.
You will be given a set of factual propositions retrieved from research papers,
and a user's question. Your job is to answer the question using ONLY the provided propositions.

STRICT RULES:
1. Every claim you make MUST be supported by at least one proposition from the list.
2. In the "claims" list, you MUST reference propositions by their ID (e.g., "prop-abc-123") in the "source_ids" array.
3. In the main "answer" field, write a clear, fluid, conversational paragraph. Do NOT include raw proposition IDs (like "prop-abc") in the main answer text. Just write naturally.
4. If the propositions don't contain enough information to answer the question, say so explicitly.
5. Do NOT invent information not present in the propositions.
6. If contradictions exist between papers, present BOTH sides - do not pick one.

Always respond with valid JSON only. No preamble, no markdown.
Output format:
{
  "answer": "A clear, natural prose answer to the question, written in complete sentences without any ugly prop-IDs.",
  "claims": [
    {
      "text": "One specific claim extracted from your answer.",
      "source_ids": ["proposition-id-1", "proposition-id-2"],
      "confidence": 0.9
    }
  ]
}"""


def _build_propositions_context(
    propositions: list[RetrievedProposition]
) -> str:
    """
    Format propositions as a numbered list for the LLM prompt.
    
    Each proposition gets:
    - Its ID (so the LLM can cite it)
    - Its source document and section (so the LLM knows the context)
    - Its text (the actual claim)
    
    Example output:
        [prop-abc-123] (Source: BERT Paper, results section, relevance: 0.91)
        "BERT-Large achieves 93.5% F1 on SQuAD 2.0."
        
        [prop-def-456] (Source: RoBERTa Paper, results section, relevance: 0.87)
        "RoBERTa achieves 94.6% F1 on SQuAD 2.0, surpassing BERT-Large."
    """
    lines = []
    for prop in propositions:
        lines.append(
            f"[{prop.proposition_id}] "
            f"(Source: '{prop.doc_title}', {prop.section_type} section, "
            f"relevance: {prop.reranker_score:.2f})\n"
            f'"{prop.text}"'
        )
    return "\n\n".join(lines)


def generate_answer(
    query: str,
    propositions: list[RetrievedProposition],
    contradictions: list = None  # list[Contradiction]
) -> tuple[str, list[Claim]]:
    """
    Generate a structured answer with per-claim source citations.
    
    Args:
        query: The user's original question.
        propositions: Top reranked propositions (should be ≤6).
        contradictions: Detected contradictions (to inform the LLM).
    
    Returns:
        A tuple of (prose_answer, claims_list).
        - prose_answer: A readable paragraph answering the question.
        - claims_list: Individual claims with source IDs and raw confidence.
                      Status (CONSENSUS/DISPUTED/etc.) NOT yet assigned here -
                      that happens in claim_classifier.py.
    """
    contradictions = contradictions or []
    
    if not propositions:
        # No evidence found - return a safe "insufficient evidence" response
        return (
            "Insufficient evidence found in the indexed documents to answer this question.",
            [Claim(
                text="Insufficient evidence found to answer this question.",
                source_proposition_ids=[],
                status=ClaimStatus.INSUFFICIENT,
                confidence=0.0
            )]
        )
    
    # ---- Build the prompt ----
    props_context = _build_propositions_context(propositions)
    
    # Add contradiction warnings if any exist
    contradiction_note = ""
    if contradictions:
        contradiction_note = (
            f"\n\nIMPORTANT: The following pairs of propositions DIRECTLY CONTRADICT each other. "
            f"You MUST present BOTH sides in your answer:\n"
        )
        for c in contradictions:
            contradiction_note += (
                f"- [{c.claim_a.proposition_id}] ('{c.claim_a.doc_title}') "
                f"CONTRADICTS [{c.claim_b.proposition_id}] ('{c.claim_b.doc_title}'): "
                f"{c.explanation}\n"
            )
    
    user_prompt = (
        f"Question: {query}\n\n"
        f"Retrieved propositions:\n{props_context}"
        f"{contradiction_note}\n\n"
        f"Answer the question using only these propositions. "
        f"Tag each claim with the proposition ID(s) that support it."
    )
    
    # ---- Call Groq ----
    try:
        client = Groq(api_key=settings.groq_api_key)
        
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Slightly above 0 for more natural prose
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content
        data = json.loads(raw)
        
    except Exception as e:
        print(f" Generation error: {e}")
        return (
            f"An error occurred during answer generation: {str(e)}\n\n(If this is a RateLimitError, wait 60 seconds for your Groq limits to reset!)",
            []
        )
    
    # ---- Parse the response ----
    prose_answer = data.get("answer", "")
    raw_claims = data.get("claims", [])
    
    # Convert raw claim dicts to Claim objects
    # Note: status is NOT set here - that's done in claim_classifier.py
    claims = []
    for raw_claim in raw_claims:
        claim = Claim(
            text=raw_claim.get("text", ""),
            source_proposition_ids=raw_claim.get("source_ids", []),
            status=ClaimStatus.SINGLE_SOURCE,  # Temporary default, will be overwritten
            confidence=float(raw_claim.get("confidence", 0.8))
        )
        claims.append(claim)
    
    print(f"   Generated answer: {len(prose_answer)} chars, {len(claims)} claims")
    
    return prose_answer, claims
