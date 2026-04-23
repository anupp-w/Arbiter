# ============================================================
# services/ingestion/chunker.py — TextBlocks → ContextChunks
# ============================================================
#
# WHAT IS CHUNKING?
# -----------------
# After the PDF parser gives us text blocks, we need to split them 
# into pieces that are the RIGHT SIZE for:
#   1. Embedding (too long = the vector loses specificity)
#   2. LLM prompts (too long = eats up context window for no reason)
#   3. Retrieval display (too long = user can't read through it)
#
# The sweet spot is ~400 tokens per chunk.
#
# ANALOGY:
# --------
# Think of it like cutting a loaf of bread:
#   - Too thin (50 tokens): Each slice is unsatisfying, no real content
#   - Just right (400 tokens): Enough for a good sandwich
#   - Too thick (2000 tokens): Can't fit in a toaster
#
# SECTION-AWARE CHUNKING:
# -----------------------
# The key innovation: we NEVER let a chunk span two sections.
# A chunk from the Results section stays in Results.
# This means if a block is 800 tokens and is all in "Results",
# we split it into two 400-token chunks, BOTH tagged as "Results".
# But we never mix "Results" text with "Methods" text in one chunk.
#
# WHY OVERLAP?
# ------------
# If we cut at exactly every 400 tokens, we might split mid-sentence:
#   Chunk 1: "...the model achieves 94.2% accuracy on"
#   Chunk 2: "MMLU, surpassing the baseline by 3%..."
#
# With 50-token overlap, Chunk 2 starts 50 tokens before the cut:
#   Chunk 1: "...the model achieves 94.2% accuracy on MMLU..."
#   Chunk 2: "...achieves 94.2% accuracy on MMLU, surpassing the baseline..."
#
# Both chunks now contain the complete claim. Overlap = insurance.
# ============================================================

import tiktoken
from pathlib import Path
from typing import Optional
import uuid

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.document import ContextChunk
from models.common import SectionType
from config import settings

# Import the TextBlock from our PDF parser
from services.ingestion.pdf_parser import TextBlock


# ============================================================
# TOKENIZER SETUP
# ============================================================
# tiktoken is OpenAI's tokenizer. We use it because:
# 1. It's fast (written in Rust under the hood)
# 2. It counts tokens the same way LLMs do
# 3. "tokens" is the right unit — not characters, not words
#
# Why tokens, not words?
# "unhappiness" is 1 word but 3 tokens: "un", "happiness" (maybe 2).
# "I" is 1 word and 1 token.
# Token count is what actually matters for LLM context windows.
# ============================================================

# "cl100k_base" is the tokenizer used by GPT-4 and similar models.
# It's close enough to Llama's tokenizer for our chunking purposes.
# (We don't need exact token counts — approximate is fine for chunking.)
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string.
    
    Example:
        >>> count_tokens("Hello, how are you?")
        5
        >>> count_tokens("The model achieves 94.2% accuracy on MMLU.")
        10
    """
    return len(_tokenizer.encode(text))


def _split_text_by_tokens(
    text: str, 
    max_tokens: int, 
    overlap_tokens: int
) -> list[str]:
    """
    Split a text into pieces of at most max_tokens, with overlap.
    
    This is the low-level splitting function. It works with token 
    indices directly for accuracy.
    
    Algorithm:
    1. Tokenize the entire text into token IDs
    2. Take slices of max_tokens length
    3. Each slice starts overlap_tokens before the previous one ended
    4. Decode each slice back into text
    
    Args:
        text: The text to split.
        max_tokens: Maximum tokens per chunk (e.g., 400).
        overlap_tokens: How many tokens to repeat between chunks (e.g., 50).
    
    Returns:
        List of text strings, each ≤ max_tokens tokens long.
    """
    # Encode the full text into token IDs
    # "Hello world" → [9906, 1917]
    tokens = _tokenizer.encode(text)
    
    # If the text is already short enough, return as-is
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Take a slice of max_tokens
        end = start + max_tokens
        
        # Decode the token slice back into text
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        
        # Move the start forward, but leave an overlap
        # If we processed tokens 0-399, next chunk starts at 350 (overlap=50)
        start = end - overlap_tokens
        
        # Safety: if overlap is >= max_tokens, we'd loop forever
        if start <= (end - max_tokens):
            start = end  # No overlap in this edge case
    
    return chunks


def create_chunks(
    text_blocks: list[TextBlock], 
    doc_id: str
) -> list[ContextChunk]:
    """
    Convert raw TextBlocks from PDF parsing into proper ContextChunks.
    
    This is the main function you'll call. It:
    1. Groups consecutive blocks from the same section
    2. Splits each group into ~400-token chunks
    3. Tags each chunk with section_type, position, page numbers
    4. Returns a list of ContextChunk objects ready for embedding
    
    Args:
        text_blocks: Output from pdf_parser.parse_pdf()
        doc_id: The document ID these chunks belong to
    
    Returns:
        List of ContextChunk objects, each ~400 tokens, 
        section-aware, with unique IDs.
    
    Example:
        >>> blocks = parse_pdf("papers/bert.pdf")
        >>> chunks = create_chunks(blocks, doc_id="abc-123")
        >>> print(len(chunks))  # e.g., 47 chunks
        >>> print(chunks[0].section_type)  # SectionType.ABSTRACT
        >>> print(chunks[0].token_count)   # ~380
    """
    if not text_blocks:
        return []
    
    # ---- Step 1: Group blocks by section ----
    # We combine consecutive blocks from the same section into one big 
    # text, then chunk that combined text. This gives us cleaner chunks 
    # than chunking each tiny block individually.
    #
    # Before grouping:
    #   [ABSTRACT block1, ABSTRACT block2, INTRO block1, INTRO block2, ...]
    # After grouping:
    #   [{"section": ABSTRACT, "text": "block1 + block2"}, 
    #    {"section": INTRO, "text": "block1 + block2"}, ...]
    
    grouped_sections: list[dict] = []
    current_group: Optional[dict] = None
    
    for block in text_blocks:
        if current_group is None or current_group["section"] != block.section_type:
            # New section starts — save current group and start a new one
            if current_group and current_group["text"].strip():
                grouped_sections.append(current_group)
            current_group = {
                "section": block.section_type,
                "text": block.text,
                "pages": [block.page_number]
            }
        else:
            # Same section — append text and track pages
            current_group["text"] += " " + block.text
            if block.page_number not in current_group["pages"]:
                current_group["pages"].append(block.page_number)
    
    # Don't forget the last group!
    if current_group and current_group["text"].strip():
        grouped_sections.append(current_group)
    
    # ---- Step 2: Split each section group into ~400-token chunks ----
    all_chunks: list[ContextChunk] = []
    total_sections = len(grouped_sections)
    
    for group_idx, group in enumerate(grouped_sections):
        section_text = group["text"]
        section_type = group["section"]
        pages = group["pages"]
        
        # Determine position in document (early/middle/late)
        # This is a simple heuristic: first third = early, middle third = middle, etc.
        if group_idx < total_sections / 3:
            position = "early"
        elif group_idx < 2 * total_sections / 3:
            position = "middle"
        else:
            position = "late"
        
        # Split the section text into token-limited chunks
        chunk_texts = _split_text_by_tokens(
            section_text,
            max_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens
        )
        
        # Create ContextChunk objects for each piece
        for chunk_text in chunk_texts:
            if chunk_text.strip():  # Skip empty chunks
                chunk = ContextChunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=chunk_text.strip(),
                    section_type=section_type,
                    token_count=count_tokens(chunk_text),
                    position=position,
                    page_numbers=pages
                )
                all_chunks.append(chunk)
    
    print(f"✅ Created {len(all_chunks)} chunks (target: ~{settings.chunk_size_tokens} tokens each)")
    
    # Print token stats for debugging
    if all_chunks:
        token_counts = [c.token_count for c in all_chunks]
        print(f"   Token range: {min(token_counts)}–{max(token_counts)}, "
              f"avg: {sum(token_counts) // len(token_counts)}")
    
    return all_chunks
