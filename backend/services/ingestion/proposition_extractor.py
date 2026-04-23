# ============================================================
# services/ingestion/proposition_extractor.py — The Core Differentiator
# ============================================================
#
# THIS IS THE MOST IMPORTANT FILE IN THE ENTIRE PROJECT.
#
# WHAT IT DOES:
# -------------
# Takes a context chunk (~400 tokens of text from a paper) and asks 
# the LLM to extract every ATOMIC FACTUAL CLAIM from it.
#
# WHAT IS AN ATOMIC FACTUAL CLAIM?
# ---------------------------------
# Think of it like breaking down a paragraph into individual Lego bricks.
# Each brick (proposition) is:
#   ✅ One single fact (not two facts combined)
#   ✅ Self-contained (makes sense without reading the rest of the paper)
#   ✅ Specific (includes numbers, names, comparisons — not vague)
#
# EXAMPLE:
# --------
# Input chunk (from a paper's Results section):
#   "Our model, which we call BERT-Large, achieves state-of-the-art 
#    results on eleven NLP benchmarks. Specifically, we observe a 
#    4.5% improvement on GLUE and a 1.5% improvement on SQuAD 2.0 
#    compared to the previous best system."
#
# Extracted propositions:
#   1. "BERT-Large achieves state-of-the-art results on eleven NLP benchmarks."
#   2. "BERT-Large achieves 4.5% improvement over previous best on GLUE."
#   3. "BERT-Large achieves 1.5% improvement over previous best on SQuAD 2.0."
#
# Notice:
#   - Each is ONE fact
#   - Each says "BERT-Large" not "our model" (self-contained)
#   - Each includes the specific number
#   - The vague "eleven NLP benchmarks" stays as-is (we can't invent detail)
#
# WHY THIS MATTERS FOR SEARCH:
# ----------------------------
# When someone asks "How does BERT perform on SQuAD?", the proposition 
# "BERT-Large achieves 1.5% improvement over previous best on SQuAD 2.0" 
# is a laser-precise answer. The original 400-token chunk would also 
# match, but it contains noise (the GLUE stuff, the vague "eleven benchmarks").
#
# Proposition indexing = precision search.
# Chunk indexing = context-rich search.
# We use BOTH (dual index) = best of both worlds.
#
# WHERE THIS TECHNIQUE COMES FROM:
# --------------------------------
# This comes from the RAPTOR paper (Recursive Abstractive Processing for 
# Tree-Organized Retrieval) and Dense Passage Retrieval literature.
# When a recruiter asks "what makes your RAG different?", this is your answer.
# ============================================================

import json
import asyncio
from pathlib import Path
from groq import Groq
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.document import Proposition, ContextChunk
from models.common import SectionType
from config import settings


# ============================================================
# THE EXTRACTION PROMPT
# ============================================================
# This is the prompt that does all the magic. It tells the LLM 
# EXACTLY what we want: atomic, self-contained, factual claims.
#
# Key prompt engineering decisions:
# 1. "Always respond with valid JSON only" — prevents markdown wrapping
# 2. Explicit good/bad examples — shows exactly what we mean
# 3. "NO opinions, NO vague statements" — filters out noise
# 4. "Include the subject explicitly" — makes propositions self-contained
#
# IMPORTANT: This prompt was iterated on extensively. If you change it,
# test on 3-4 different paper chunks to make sure it still works.
# Bad prompts = bad propositions = bad search = bad answers = bad project.
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """You are a precise factual claim extraction system.
Your job is to extract EVERY atomic factual claim from the given text.

RULES:
1. Each claim must be a SINGLE fact — never combine two facts into one sentence.
2. Each claim must be SELF-CONTAINED — it must make complete sense without reading any other text. Replace pronouns ("it", "they", "our model") with the actual entity name.
3. Each claim must be SPECIFIC — include numbers, model names, dataset names, and comparisons when present.
4. Extract ONLY factual claims — NO opinions, NO vague statements, NO references to figures/tables.
5. If the text contains no factual claims (e.g., it's just an outline or table of contents), return an empty list.

GOOD examples (what you should produce):
- "BERT-Large achieves 93.5% F1 score on the SQuAD 2.0 benchmark."
- "The Transformer architecture replaces recurrence with multi-head self-attention."
- "Pre-training on 16GB of text data improves GLUE score by 4.5% compared to training from scratch."

BAD examples (what you should NOT produce):
- "The results are shown in Table 3." (reference, not a claim)
- "It achieves good results." (vague, uses pronoun)
- "The model is fast and accurate and outperforms everything." (multiple claims, vague)
- "We believe this approach is promising." (opinion)

Always respond with valid JSON only. No preamble, no explanation, no markdown.
Output format: {"propositions": ["claim one", "claim two", ...]}"""


def _create_groq_client() -> Groq:
    """
    Create a Groq API client.
    
    Groq's API is almost identical to OpenAI's API — same concept, 
    same method names, just pointing to Groq's servers instead.
    The benefit: Groq runs Llama 3.3 70B at incredible speed 
    (sometimes 10x faster than OpenAI) and the free tier is generous.
    """
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set! Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com/keys"
        )
    return Groq(api_key=settings.groq_api_key)


def extract_propositions_from_chunk(
    chunk: ContextChunk,
    doc_title: str = "",
    max_retries: int = 2
) -> list[Proposition]:
    """
    Extract atomic factual propositions from a single context chunk.
    
    This is where we call the LLM and ask it to decompose the chunk 
    into individual facts. The LLM returns JSON, we parse it, and 
    create Proposition objects.
    
    RETRY LOGIC:
    LLMs sometimes fail to produce valid JSON (they might add "```json" 
    markers or extra text). We retry up to max_retries times. If all 
    retries fail, we return an empty list rather than crashing.
    
    Args:
        chunk: The ContextChunk to extract propositions from.
        doc_title: Title of the document (used to fill in the doc_title field).
        max_retries: How many times to retry if JSON parsing fails.
    
    Returns:
        List of Proposition objects, each containing one atomic claim.
    
    Example:
        >>> chunk = ContextChunk(doc_id="123", text="BERT achieves 93.5% on SQuAD...")
        >>> props = extract_propositions_from_chunk(chunk, doc_title="BERT Paper")
        >>> print(props[0].text)
        "BERT achieves 93.5% F1 score on SQuAD 2.0."
    """
    client = _create_groq_client()
    
    # ---- Build the user prompt ----
    # We tell the LLM what section the text is from, because:
    # - Results section → likely has concrete numbers (high-value propositions)
    # - Introduction → likely has general background claims (lower priority)
    # - Methods → likely has technical details about the approach
    user_prompt = (
        f"Extract all atomic factual claims from this {chunk.section_type.value} section text:\n\n"
        f"{chunk.text}"
    )
    
    # ---- Call the LLM with retries ----
    for attempt in range(max_retries + 1):
        try:
            # This is the actual API call to Groq.
            # It's identical to the OpenAI API format:
            #   client.chat.completions.create(messages=[...])
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                # temperature=0.0 means "be as deterministic as possible"
                # We want consistent, factual extraction — no creativity.
                temperature=0.0,
                # max_tokens limits the response length.
                # 2000 is enough for ~30 propositions (more than any chunk will have).
                max_tokens=2000,
                # response_format tells Groq to output valid JSON.
                # This significantly reduces JSON parsing failures.
                response_format={"type": "json_object"}
            )
            
            # ---- Parse the JSON response ----
            raw_content = response.choices[0].message.content
            
            # Sometimes the LLM wraps JSON in ```json ... ``` markers.
            # We strip those just in case.
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                # Remove markdown code fences
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            
            # Parse the JSON
            data = json.loads(cleaned)
            
            # Extract the list of proposition texts
            prop_texts = data.get("propositions", [])
            
            # ---- Create Proposition objects ----
            propositions = []
            for prop_text in prop_texts:
                # Skip empty strings or very short "propositions"
                if not prop_text or len(prop_text.strip()) < 10:
                    continue
                
                prop = Proposition(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.id,
                    text=prop_text.strip(),
                    section_type=chunk.section_type,
                    extraction_confidence=1.0,  # We trust the LLM's extraction
                    doc_title=doc_title
                )
                propositions.append(prop)
            
            return propositions
            
        except json.JSONDecodeError as e:
            # The LLM didn't produce valid JSON. Retry.
            print(f"⚠️  JSON parse error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                print(f"   Retrying... ({attempt + 2}/{max_retries + 1})")
                continue
            else:
                print(f"❌ Failed to parse JSON after {max_retries + 1} attempts. Skipping chunk.")
                return []
                
        except Exception as e:
            # Something else went wrong (network error, rate limit, etc.)
            print(f"❌ Error extracting propositions: {e}")
            if attempt < max_retries:
                # Wait a bit before retrying (exponential backoff)
                import time
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                return []
    
    return []  # Should never reach here, but just in case


def extract_propositions_from_chunks(
    chunks: list[ContextChunk],
    doc_title: str = ""
) -> list[Proposition]:
    """
    Extract propositions from ALL chunks in a document.
    
    This processes chunks ONE AT A TIME (not in parallel) because:
    1. Groq's free tier has rate limits (~30 requests/minute)
    2. Sequential processing is easier to debug
    3. The total time is still reasonable (~2-5 minutes for a 30-page paper)
    
    For a production system, you'd batch these with rate limiting.
    For a portfolio project, sequential is fine.
    
    Args:
        chunks: List of all ContextChunks from a document.
        doc_title: Title of the source document.
    
    Returns:
        List of ALL propositions extracted from ALL chunks.
    """
    all_propositions: list[Proposition] = []
    
    print(f"\n🔬 Extracting propositions from {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i + 1}/{len(chunks)} [{chunk.section_type.value}]...", end=" ")
        
        props = extract_propositions_from_chunk(chunk, doc_title=doc_title)
        all_propositions.extend(props)
        
        print(f"→ {len(props)} propositions")
        
        # Small delay to respect Groq's rate limits.
        # Free tier allows ~30 requests/minute = 1 request every 2 seconds.
        # We wait 1.5s to stay safely under the limit.
        import time
        time.sleep(1.5)
    
    print(f"\n✅ Total: {len(all_propositions)} propositions from {len(chunks)} chunks")
    
    return all_propositions
