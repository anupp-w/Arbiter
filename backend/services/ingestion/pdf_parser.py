# ============================================================
# services/ingestion/pdf_parser.py — PDF → Structured Text Blocks
# ============================================================
#
# WHAT THIS FILE DOES:
# --------------------
# Takes a PDF file and extracts text from it, BUT — and this is key —
# it tries to PRESERVE the document structure. Instead of dumping all 
# text into one giant blob, we detect:
#   - Section headers (Abstract, Introduction, Methods, etc.)
#   - What section each paragraph belongs to
#   - Page numbers for each block of text
#
# WHY STRUCTURE MATTERS:
# ----------------------
# Imagine you're searching for "accuracy results" in a paper.
# A claim from the Results section is MORE authoritative than 
# the same claim mentioned in passing in the Introduction.
# By preserving structure, we can use section_type as a quality signal.
#
# HOW WE DETECT SECTIONS:
# -----------------------
# We use simple heuristics (rules of thumb), NOT a machine learning model:
#   1. Look for lines that match known section names 
#      (e.g., "Abstract", "1. Introduction", "4 Results")
#   2. Short lines followed by longer paragraphs are likely headers
#   3. Lines in ALL CAPS or with numbering patterns are likely headers
#
# Is this perfect? No. But it's MUCH better than treating all text 
# the same, and the imperfection is an honest thing to discuss 
# in your README failure modes section.
#
# PDFPLUMBER:
# -----------
# pdfplumber is a Python library that reads PDFs and gives you text 
# with positioning information. Unlike PyPDF2 (which gives you raw text),
# pdfplumber preserves layout information that helps us detect structure.
# ============================================================

import pdfplumber
import re
from pathlib import Path
from dataclasses import dataclass

# We import our enum that defines valid section types
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.common import SectionType


@dataclass
class TextBlock:
    """
    A block of text from the PDF with its metadata.
    
    This is an INTERMEDIATE object — it exists between "raw PDF text" 
    and our final ContextChunk model. Think of it as raw material 
    that the chunker will later process.
    
    We use @dataclass instead of Pydantic here because this is 
    internal-only — it never goes to the API or UI. Dataclasses 
    are simpler and faster for internal data passing.
    """
    text: str                    # The actual text content
    section_type: SectionType    # Which section this belongs to
    page_number: int             # Which PDF page it came from
    is_header: bool = False      # Is this a section header itself?


# ============================================================
# SECTION DETECTION PATTERNS
# ============================================================
# These regex patterns detect common section header formats in 
# academic papers. Each pattern tries to match a different style:
#
# "Abstract"           → just the word
# "1. Introduction"    → numbered with period
# "1 Introduction"     → numbered without period
# "2.1 Related Work"   → sub-numbered
# "METHODS"            → all caps
# "IV. Results"        → Roman numerals
#
# We map each detected header to our SectionType enum.
# ============================================================

# Maps common header text (lowercase) to our SectionType enum
SECTION_KEYWORDS: dict[str, SectionType] = {
    "abstract": SectionType.ABSTRACT,
    "introduction": SectionType.INTRODUCTION,
    "related work": SectionType.RELATED_WORK,
    "related works": SectionType.RELATED_WORK,
    "background": SectionType.RELATED_WORK,
    "prior work": SectionType.RELATED_WORK,
    "literature review": SectionType.RELATED_WORK,
    "method": SectionType.METHODS,
    "methods": SectionType.METHODS,
    "methodology": SectionType.METHODS,
    "approach": SectionType.METHODS,
    "model": SectionType.METHODS,
    "proposed method": SectionType.METHODS,
    "architecture": SectionType.METHODS,
    "setup": SectionType.METHODS,
    "experimental setup": SectionType.METHODS,
    "experiment": SectionType.RESULTS,
    "experiments": SectionType.RESULTS,
    "results": SectionType.RESULTS,
    "evaluation": SectionType.RESULTS,
    "findings": SectionType.RESULTS,
    "analysis": SectionType.RESULTS,
    "discussion": SectionType.DISCUSSION,
    "limitations": SectionType.DISCUSSION,
    "conclusion": SectionType.CONCLUSION,
    "conclusions": SectionType.CONCLUSION,
    "summary": SectionType.CONCLUSION,
    "future work": SectionType.CONCLUSION,
    "concluding remarks": SectionType.CONCLUSION,
}


def _clean_text(text: str) -> str:
    """
    Clean up raw PDF text by removing common artifacts.
    
    PDFs often contain:
    - Multiple consecutive spaces (from column layouts)
    - Hyphenated line breaks (from justified text: "impor-\\ntant")
    - Non-breaking spaces and other unicode weirdness
    
    This function normalizes all of that into clean, readable text.
    """
    # Replace hyphenated line breaks: "impor-\ntant" → "important"
    # This happens because PDFs justify text by breaking words across lines.
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # Replace remaining newlines with spaces (within a paragraph)
    text = re.sub(r'\n', ' ', text)
    
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()


def _detect_section_type(line: str) -> SectionType | None:
    """
    Try to identify if a line is a section header, and if so, which section.
    
    Returns the SectionType if detected, or None if this isn't a header.
    
    Examples:
        _detect_section_type("1. Introduction")  → SectionType.INTRODUCTION
        _detect_section_type("ABSTRACT")          → SectionType.ABSTRACT
        _detect_section_type("The cat sat down")  → None (not a header)
    """
    # Clean up the line for matching
    cleaned = line.strip()
    
    # Skip empty lines or very long lines (headers are usually short)
    if not cleaned or len(cleaned) > 80:
        return None
    
    # Remove common numbering prefixes:
    # "1. ", "1 ", "2.1 ", "2.1. ", "IV. ", "A. "
    stripped = re.sub(
        r'^[\dIVXivx]+[\.\)]\s*'   # "1. " or "IV. " or "1) "
        r'|^[\d]+\.[\d]+\.?\s*'     # "2.1 " or "2.1. "
        r'|^[A-Z][\.\)]\s*',       # "A. " or "A) "
        '',
        cleaned
    ).strip()
    
    # Now check if the remaining text matches any known section keyword
    stripped_lower = stripped.lower()
    
    for keyword, section_type in SECTION_KEYWORDS.items():
        # Check for exact match or close match
        if stripped_lower == keyword:
            return section_type
        # Also check if the keyword appears at the start
        # (handles "Introduction and Background" → INTRODUCTION)
        if stripped_lower.startswith(keyword) and len(stripped_lower) < len(keyword) + 20:
            return section_type
    
    # Check if the entire line is uppercase (common for headers like "METHODS")
    if cleaned.isupper() and len(cleaned.split()) <= 5:
        # It's uppercase and short — probably a header
        for keyword, section_type in SECTION_KEYWORDS.items():
            if keyword in cleaned.lower():
                return section_type
    
    return None


def parse_pdf(pdf_path: str | Path) -> list[TextBlock]:
    """
    Parse a PDF file into structured text blocks.
    
    This is the ENTRY POINT for the ingestion pipeline. It takes a 
    PDF file path and returns a list of TextBlock objects, each tagged 
    with its section type and page number.
    
    The algorithm:
    1. Open PDF with pdfplumber
    2. Extract text page by page
    3. Split each page into paragraphs (double newline separated)
    4. For each paragraph, check if it's a section header
    5. Track the "current section" and tag all text accordingly
    
    Args:
        pdf_path: Path to the PDF file.
    
    Returns:
        List of TextBlock objects, in reading order, each tagged 
        with section_type, page_number, and is_header flag.
    
    Example:
        >>> blocks = parse_pdf("papers/bert.pdf")
        >>> for b in blocks[:3]:
        ...     print(f"[{b.section_type.value}] {b.text[:60]}...")
        [abstract] We introduce a new language representation model...
        [introduction] Language model pre-training has been shown...
        [introduction] There are two existing strategies for apply...
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    blocks: list[TextBlock] = []
    
    # Track which section we're currently in.
    # Start with ABSTRACT because most papers begin there.
    # If no section header is ever detected, everything stays ABSTRACT/OTHER.
    current_section = SectionType.ABSTRACT
    
    # ---- Open and extract text from each page ----
    with pdfplumber.open(str(pdf_path)) as pdf:
        print(f"📄 Parsing PDF: {pdf_path.name} ({len(pdf.pages)} pages)")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract raw text from this page
            raw_text = page.extract_text()
            
            if not raw_text:
                # Some pages are purely images/figures — skip them
                continue
            
            # Split into paragraphs using double newlines or large gaps.
            # Single newlines within a paragraph are just line wraps.
            paragraphs = re.split(r'\n{2,}', raw_text)
            
            for para in paragraphs:
                # Split paragraph into lines to check for headers
                lines = para.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # ---- Check if this line is a section header ----
                    detected_section = _detect_section_type(line)
                    
                    if detected_section is not None:
                        # We found a new section! Update current_section.
                        current_section = detected_section
                        blocks.append(TextBlock(
                            text=line,
                            section_type=current_section,
                            page_number=page_num,
                            is_header=True
                        ))
                    else:
                        # Regular text — tag it with current section
                        cleaned = _clean_text(line)
                        if len(cleaned) > 20:  # Skip very short lines (page numbers, etc.)
                            blocks.append(TextBlock(
                                text=cleaned,
                                section_type=current_section,
                                page_number=page_num,
                                is_header=False
                            ))
    
    # ---- Post-processing: merge tiny consecutive blocks from same section ----
    # PDF extraction often splits what should be one paragraph into multiple 
    # lines. We merge consecutive non-header blocks from the same section.
    merged_blocks: list[TextBlock] = []
    
    for block in blocks:
        if block.is_header:
            # Headers stay separate — they're structural markers
            merged_blocks.append(block)
        elif (
            merged_blocks 
            and not merged_blocks[-1].is_header 
            and merged_blocks[-1].section_type == block.section_type
            and merged_blocks[-1].page_number == block.page_number
        ):
            # Same section, same page, both are text → merge them
            merged_blocks[-1].text += " " + block.text
        else:
            # Different section or different page → new block
            merged_blocks.append(block)
    
    # Filter out header-only blocks (we don't need to search through them)
    content_blocks = [b for b in merged_blocks if not b.is_header]
    
    print(f"✅ Extracted {len(content_blocks)} text blocks from {pdf_path.name}")
    
    # Print a summary of sections found
    section_counts: dict[str, int] = {}
    for b in content_blocks:
        section_counts[b.section_type.value] = section_counts.get(b.section_type.value, 0) + 1
    print(f"   Sections found: {section_counts}")
    
    return content_blocks


def extract_title_from_pdf(pdf_path: str | Path) -> str:
    """
    Try to extract the paper's title from the PDF.
    
    Strategy:
    1. Try PDF metadata (some PDFs have title in metadata)
    2. Fall back to first line of text (usually the title in academic papers)
    3. Fall back to filename
    
    This is a best-effort extraction. Getting it wrong is fine —
    the user can always edit the title.
    """
    pdf_path = Path(pdf_path)
    
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            # Strategy 1: PDF metadata
            metadata = pdf.metadata or {}
            if metadata.get("Title") and len(metadata["Title"].strip()) > 5:
                return metadata["Title"].strip()
            
            # Strategy 2: First page, first significant line
            if pdf.pages:
                first_page_text = pdf.pages[0].extract_text() or ""
                lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
                # Skip very short lines (page numbers, headers like "arXiv:xxxx")
                for line in lines:
                    if len(line) > 10 and not line.startswith("arXiv"):
                        return line[:200]  # Cap at 200 chars just in case
    except Exception:
        pass
    
    # Strategy 3: Filename without extension
    return pdf_path.stem.replace('_', ' ').replace('-', ' ').title()
