# ============================================================
# models/common.py — Shared Enums and Base Types
# ============================================================
#
# WHAT ARE ENUMS?
# ---------------
# An Enum is a fixed set of choices. Like a dropdown menu in a form.
# Instead of using raw strings like "processing" or "completed" 
# (which you might typo as "complted"), you use DocumentStatus.COMPLETED.
# The IDE autocompletes it, and if you typo, Python yells immediately.
#
# Think of it like this:
#   Without enums: status = "procesing"  ← typo, no error, silent bug
#   With enums:    status = DocumentStatus.PROCESING  ← Python crashes immediately
#
# WHAT'S IN THIS FILE:
# --------------------
# All the "vocabulary" that multiple parts of the system share.
# Document statuses, section types, claim statuses, etc.
# ============================================================

from enum import Enum


class DocumentStatus(str, Enum):
    """
    Every document goes through these stages during ingestion.
    
    Think of it like an order status on Amazon:
    PENDING    → "We received your order" (PDF uploaded, waiting to process)
    PROCESSING → "Your order is being prepared" (extracting text, creating embeddings)
    COMPLETED  → "Delivered!" (fully indexed, ready for queries)
    FAILED     → "Delivery failed" (something went wrong during processing)
    
    The (str, Enum) part means this enum's values ARE strings.
    So DocumentStatus.PENDING == "pending" is True.
    This matters when we serialize to JSON for the API.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SectionType(str, Enum):
    """
    What part of the paper a chunk of text came from.
    
    This matters because a claim from the Results section is more 
    authoritative than the same claim in the Introduction.
    For example:
    - "Our model achieves 95% accuracy" in RESULTS → actual finding
    - "Our model achieves 95% accuracy" in ABSTRACT → summary (still good)
    - "Prior work achieved 95% accuracy" in INTRODUCTION → someone else's claim
    
    We detect these during PDF parsing using heuristics (font size, 
    position, keyword matching). It's not perfect, but even rough 
    section detection is WAY better than treating all text equally.
    """
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    OTHER = "other"


class ClaimStatus(str, Enum):
    """
    The THREE claim statuses — this is the entire identity of Arbiter.
    
    Every claim in our generated answer gets exactly one of these labels:
    
    CONSENSUS     → 🟢 Multiple papers agree on this claim.
                     Example: "Transformers outperform RNNs on NLP tasks"
                     (BERT, GPT-3, and RoBERTa all say this)
    
    DISPUTED      → 🔴 Papers DISAGREE on this claim. Both sides shown.
                     Example: "Model size matters more than data size"
                     (Kaplan says yes, Chinchilla says no)
                     THIS IS THE DEMO MOMENT. When a recruiter sees a 
                     red badge with both sides of a disagreement shown 
                     with their sources — that's the wow factor.
    
    SINGLE_SOURCE → 🟡 Only ONE paper mentions this. Could be true, 
                     but we can't cross-verify it.
                     Example: "RAG reduces hallucination by 30%"
                     (Only the RAG paper says this, no confirmation)
    
    INSUFFICIENT  → ⚪ We couldn't find enough evidence to answer.
                     This is better than hallucinating an answer.
    """
    CONSENSUS = "consensus"
    DISPUTED = "disputed"
    SINGLE_SOURCE = "single_source"
    INSUFFICIENT = "insufficient"


class PropositionRelationship(str, Enum):
    """
    When we compare two propositions, their relationship is one of:
    
    SUPPORT    → They agree. "A says X" and "B also says X."
    CONTRADICT → They disagree. "A says X" but "B says NOT X."
    COMPLEMENT → They cover different aspects. "A talks about speed, B about accuracy."
    UNRELATED  → They don't address the same topic at all.
    
    We use this in the pairwise contradiction detection step (Phase 3).
    The LLM classifies every pair of propositions into one of these.
    """
    SUPPORT = "support"
    CONTRADICT = "contradict"
    COMPLEMENT = "complement"
    UNRELATED = "unrelated"
