# ============================================================
# frontend/app.py - Streamlit UI (Main Entry Point)
# ============================================================
#
# WHAT IS STREAMLIT?
# ------------------
# Streamlit is a Python library that turns Python scripts into 
# web applications. You write Python code, and it renders as 
# a beautiful web UI. No HTML, no CSS, no JavaScript needed.
#
# HOW IT WORKS:
# -------------
# Streamlit re-runs your ENTIRE script from top to bottom every 
# time the user interacts with anything (clicks a button, types text).
# This might sound inefficient, but Streamlit caches heavy operations
# (like loading models) using @st.cache_resource.
#
# ARCHITECTURE:
# -------------
# This file can work in TWO modes:
#
# Mode 1: LOCAL DEVELOPMENT
#   Streamlit calls the FastAPI backend via HTTP requests.
#   Run both servers: FastAPI on port 8000, Streamlit on port 8501.
#
# Mode 2: STREAMLIT CLOUD (production)
#   Imports the service functions DIRECTLY instead of HTTP calls.
#   Streamlit Cloud doesn't have a separate FastAPI server.
#   We detect which mode we're in and switch automatically.
#
# The UI has three sections:
#   LEFT SIDEBAR  → Document management (upload, status, list)
#   CENTER/MAIN   → Query input
#   MAIN AREA     → Structured answer with colored claim badges
# ============================================================

import streamlit as st
import sys
from pathlib import Path
import time
import json

# ---- Page Configuration ----
# This MUST be the first Streamlit command in the file.
# It sets the browser tab title, icon, and layout.
st.set_page_config(
    page_title="Arbiter - Research Synthesizer",
    page_icon="",
    layout="wide",  # Use full browser width
    initial_sidebar_state="expanded"
)

# ============================================================
# BACKEND IMPORT STRATEGY
# ============================================================
# We try to import the backend services directly (for Streamlit Cloud).
# If that fails (e.g., different directory structure), we fall back
# to HTTP calls to the FastAPI server.
# ============================================================

# Add the backend directory to Python path
# This allows Streamlit to import backend modules directly
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    # Try direct import (works on Streamlit Cloud and local)
    from services.ingestion.pipeline import (
        create_document_record,
        run_ingestion_pipeline,
        get_document,
        get_all_documents
    )
    from services.retrieval.embedder import get_embedder
    from services.retrieval.faiss_store import get_proposition_store, get_chunk_store
    from services.retrieval.bm25_store import get_bm25_store
    from services.retrieval.hybrid import hybrid_retrieve
    from services.retrieval.reranker import get_reranker
    from services.analysis.contradiction import detect_contradictions
    from services.analysis.claim_classifier import classify_all_claims
    from services.generation.answer_generator import generate_answer
    from services.generation.post_processor import verify_sources, compute_confidence
    from models.query import QueryRequest, QueryResult, RetrievedProposition
    from models.common import ClaimStatus, DocumentStatus
    from config import settings
    
    # FORCE the model to 8B to bypass Streamlit's aggressive module caching!
    settings.groq_model = "llama-3.1-8b-instant"
    
    DIRECT_MODE = True

except ImportError as e:
    # Fall back to HTTP mode (calls FastAPI server)
    import requests
    DIRECT_MODE = False
    BACKEND_URL = "http://localhost:8000"


# ============================================================
# CUSTOM CSS - Make the badges look beautiful
# ============================================================
# Streamlit's default styling is fine but we want the claim 
# badges (Consensus/Disputed/Single-Source) to really POP.
# We inject custom CSS using st.markdown with unsafe_allow_html=True.
# ============================================================

st.markdown("""
<style>
    /* ---- Badge styles ---- */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    
    .badge-consensus {
        background-color: #1a472a;
        color: #69db7c;
        border: 1px solid #2f9e44;
    }
    
    .badge-disputed {
        background-color: #7a1414;
        color: #ff8787;
        border: 1px solid #e03131;
    }
    
    .badge-single {
        background-color: #7d4e1a;
        color: #ffd43b;
        border: 1px solid #e67700;
    }
    
    .badge-insufficient {
        background-color: #2c2c2c;
        color: #adb5bd;
        border: 1px solid #495057;
    }
    
    .badge-warning {
        background-color: #4a1942;
        color: #f783ac;
        border: 1px solid #a61e4d;
    }

    /* ---- Claim card styles ---- */
    .claim-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-left: 4px solid #444;
    }
    
    .claim-card-consensus { border-left-color: #2f9e44; }
    .claim-card-disputed  { border-left-color: #e03131; }
    .claim-card-single    { border-left-color: #e67700; }
    
    .claim-text {
        font-size: 15px;
        color: #e0e0e0;
        line-height: 1.6;
        margin-top: 6px;
    }
    
    .source-label {
        font-size: 11px;
        color: #888;
        margin-top: 8px;
    }

    /* ---- Confidence bar label ---- */
    .conf-label {
        font-size: 13px;
        color: #aaa;
        margin-top: 4px;
    }
    
    /* ---- Contradiction box ---- */
    .contradiction-box {
        background: #1a0a0a;
        border: 1px solid #7a1414;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
    }
    
    .contradiction-side {
        background: #2a1a1a;
        border-radius: 6px;
        padding: 10px;
        margin: 6px 0;
        font-size: 13px;
        color: #ddd;
    }
    
    /* ---- Main answer box ---- */
    .answer-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        font-size: 16px;
        line-height: 1.7;
        color: #e0e0e0;
    }
    
    /* ---- Section headers ---- */
    .section-header {
        font-size: 14px;
        font-weight: 700;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 24px 0 12px 0;
        border-bottom: 1px solid #333;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED MODEL LOADERS
# ============================================================
# @st.cache_resource tells Streamlit:
# "Run this function ONCE, cache the result, and reuse it 
# every time the app re-runs."
# Without this, the embedding model would reload on EVERY user click.
# With this, it loads ONCE and stays in memory. Critical for performance.
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    """Load the embedding model once and cache it."""
    return get_embedder()

@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker():
    """Load the reranker once and cache it."""
    return get_reranker()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_badge_html(status: str) -> str:
    """Return the HTML for a colored status badge."""
    status_map = {
        "consensus":    ("badge-consensus", "Consensus"),
        "disputed":     ("badge-disputed",  "Disputed"),
        "single_source":("badge-single",    "Single Source"),
        "insufficient": ("badge-insufficient", "Insufficient"),
    }
    css_class, label = status_map.get(status, ("badge-insufficient", "Unknown"))
    return f'<span class="badge {css_class}">{label}</span>'


def render_claim_card(claim: dict, prop_lookup: dict) -> None:
    """
    Render a single claim as a styled card with its badge.
    
    For DISPUTED claims, expand to show both sides of the contradiction.
    For all claims, show the source document(s).
    """
    status = claim.get("status", "insufficient")
    text = claim.get("text", "")
    source_ids = claim.get("source_proposition_ids", [])
    contradiction = claim.get("contradiction")
    
    # Determine card border color class
    card_class_map = {
        "consensus":     "claim-card-consensus",
        "disputed":      "claim-card-disputed",
        "single_source": "claim-card-single",
    }
    card_class = card_class_map.get(status, "")
    
    # Get source document titles
    source_docs = list({
        prop_lookup.get(pid, {}).get("doc_title", "Unknown Paper")
        for pid in source_ids
        if pid in prop_lookup
    })
    source_text = ", ".join(source_docs) if source_docs else "No verified source"
    
    # Render the card
    st.markdown(
        f"""<div class="claim-card {card_class}">
            {get_badge_html(status)}
            <div class="claim-text">{text}</div>
            <div class="source-label"> {source_text}</div>
        </div>""",
        unsafe_allow_html=True
    )
    
    # For DISPUTED claims, show the contradiction details in an expander
    if status == "disputed" and contradiction:
        with st.expander(" See both sides of this disagreement"):
            st.markdown(
                '<div class="contradiction-box">'
                f'<div style="font-size:12px; color:#ff8787; font-weight:700; margin-bottom:8px;">'
                f' {contradiction.get("explanation", "Direct factual conflict detected")}</div>',
                unsafe_allow_html=True
            )
            
            col_a, col_b = st.columns(2)
            
            claim_a = contradiction.get("claim_a", {})
            claim_b = contradiction.get("claim_b", {})
            
            with col_a:
                st.markdown(
                    f'<div class="contradiction-side">'
                    f'<strong style="color:#ff8787;">Paper A: {claim_a.get("doc_title", "")}</strong><br>'
                    f'<em>({claim_a.get("section_type", "")} section)</em><br><br>'
                    f'"{claim_a.get("text", "")}"'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with col_b:
                st.markdown(
                    f'<div class="contradiction-side">'
                    f'<strong style="color:#ff8787;">Paper B: {claim_b.get("doc_title", "")}</strong><br>'
                    f'<em>({claim_b.get("section_type", "")} section)</em><br><br>'
                    f'"{claim_b.get("text", "")}"'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR - Document Management
# ============================================================

def render_sidebar():
    """Render the left sidebar with document management."""
    
    with st.sidebar:
        st.markdown("##  Arbiter")
        st.markdown("*Multi-document research synthesizer*")
        st.divider()
        
        # ---- Upload Section ----
        st.markdown("###  Upload Paper")
        
        uploaded_file = st.file_uploader(
            "Drop a PDF research paper",
            type=["pdf"],
            help="Upload academic papers in PDF format. Ingestion takes 2-5 minutes."
        )
        
        if uploaded_file:
            if st.button(" Ingest Paper", type="primary", use_container_width=True):
                _ingest_document(uploaded_file)
        
        st.divider()
        
        # ---- Document List ----
        st.markdown("###  Indexed Papers")
        
        if DIRECT_MODE:
            # Convert Pydantic models to dicts so they match the HTTP JSON format
            docs = [d.model_dump(mode="json") for d in get_all_documents()]
        else:
            try:
                resp = requests.get(f"{BACKEND_URL}/documents", timeout=5)
                docs = resp.json() if resp.ok else []
            except Exception:
                docs = []
        
        if not docs:
            st.info("No papers indexed yet. Upload a PDF above to start.")
        else:
            for doc in docs:
                _render_doc_status_card(doc)
        
        st.divider()
        
        # ---- Stats ----
        if DIRECT_MODE:
            prop_count = get_proposition_store().count
            chunk_count = get_chunk_store().count
            
            st.markdown("###  Index Stats")
            col1, col2 = st.columns(2)
            col1.metric("Propositions", prop_count)
            col2.metric("Chunks", chunk_count)


def _render_doc_status_card(doc: dict) -> None:
    """Render a small status card for each document in the sidebar."""
    status = doc.get("status", "unknown")
    title = doc.get("title", "Unknown")[:35]  # Truncate long titles
    
    # Status icon
    icon_map = {
        "completed": "",
        "processing": "",
        "pending": "",
        "failed": ""
    }
    icon = icon_map.get(status, "")
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"**{title}**")
        col2.markdown(f"{icon}")
        
        if status == "completed":
            props = doc.get("num_propositions", 0)
            st.caption(f"{props} propositions indexed")
        elif status == "processing":
            st.caption("Processing... (refresh to update)")
        elif status == "failed":
            st.caption(f" Failed: {doc.get('error_message', 'Unknown error')[:50]}")
        
        st.markdown("---")


def _ingest_document(uploaded_file) -> None:
    """Handle document upload and ingestion."""
    import tempfile, shutil, asyncio
    
    with st.spinner(f"Uploading '{uploaded_file.name}'..."):
        if DIRECT_MODE:
            # Save to temp file first
            settings.documents_dir.mkdir(parents=True, exist_ok=True)
            save_path = settings.documents_dir / uploaded_file.name
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            doc = create_document_record(
                filename=uploaded_file.name,
                pdf_path=str(save_path)
            )
            
            st.success(f" '{doc.title}' queued for processing!")
            st.info(
                f"Ingestion runs in the background. "
                f"Refresh the page to see progress. "
                f"It takes ~3-5 minutes per paper."
            )
            
            # In Streamlit, we can't truly run background tasks.
            # We run the pipeline synchronously with a progress indicator.
            # For production, use FastAPI's BackgroundTasks instead.
            with st.spinner("Ingesting paper (this takes 3-5 minutes)..."):
                try:
                    asyncio.run(run_ingestion_pipeline(
                        doc_id=doc.id,
                        pdf_path=str(save_path)
                    ))
                    st.success(f" '{doc.title}' fully ingested and ready to query!")
                    st.rerun()
                except Exception as e:
                    st.error(f" Ingestion failed: {str(e)}")
        else:
            # HTTP mode - send to FastAPI
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/documents",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=30
                )
                if resp.ok:
                    data = resp.json()
                    st.success(f" Paper queued! ID: {data['doc_id']}")
                else:
                    st.error(f" Upload failed: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(" Cannot connect to backend. Is FastAPI running on port 8000?")


# ============================================================
# MAIN AREA - Query + Results
# ============================================================

def render_main_area():
    """Render the main query area and results panel."""
    
    st.markdown("#  Arbiter")
    st.markdown(
        "*Ask a question across your indexed research papers. "
        "Every claim is labeled: **Consensus**, **Disputed**, or **Single-Source**.*"
    )
    st.divider()
    
    # ---- Query Input ----
    query = st.text_input(
        "Research Question",
        placeholder="e.g. How do different papers approach the scaling of model size?",
        help="Ask any research question. Arbiter will retrieve evidence from all indexed papers.",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button(" Ask Arbiter", type="primary", use_container_width=True)
    
    if submit and query.strip():
        with st.spinner("Retrieving and synthesizing... (this takes ~10 seconds)"):
            result = _run_query(query.strip())
        
        if result:
            _render_results(result)
    elif submit and not query.strip():
        st.warning("Please enter a research question.")
    
    # ---- Example Queries ----
    st.markdown("---")
    st.markdown("** Try these example queries:**")
    
    examples = [
        "How do different papers approach scaling of model size vs data size?",
        "What do papers say about the performance of pre-trained language models?",
        "What are the key differences in training approaches across papers?",
    ]
    
    for example in examples:
        if st.button(f"→ {example}", key=f"ex_{example[:20]}"):
            # Set the query input and trigger re-run
            st.session_state.query_input = example
            st.rerun()


def _run_query(query: str) -> dict | None:
    """Run the full query pipeline and return the result dict."""
    
    if DIRECT_MODE:
        # Direct import mode - call services directly
        import asyncio
        
        try:
            # Load cached models
            load_embedder()
            load_reranker()
            
            # Run the async hybrid retrieval
            raw_results = asyncio.run(hybrid_retrieve(query))
            
            if not raw_results:
                st.warning(" No relevant documents found. Have you uploaded any papers?")
                return None
            
            # Rerank
            reranker = load_reranker()
            top_results = reranker.rerank(query, raw_results, top_k=settings.top_k_rerank)
            
            # Convert to RetrievedProposition objects
            props = []
            for r in top_results:
                props.append(RetrievedProposition(
                    proposition_id=r.get("id", ""),
                    text=r.get("text", ""),
                    doc_id=r.get("doc_id", ""),
                    doc_title=r.get("doc_title", "Unknown"),
                    chunk_id=r.get("chunk_id", ""),
                    section_type=r.get("section_type", "other"),
                    rrf_score=r.get("rrf_score", 0.0),
                    reranker_score=r.get("reranker_score", 0.0)
                ))
            
            # Contradiction detection
            contradictions = detect_contradictions(props)
            
            # Generation
            prose, raw_claims = generate_answer(query, props, contradictions)
            
            # Post-processing
            verified = verify_sources(raw_claims, props)
            classified = classify_all_claims(verified, props, contradictions)
            confidence = compute_confidence(classified, props)
            
            # Build result dict
            return {
                "query_text": query,
                "main_answer": prose,
                "claims": [c.model_dump(mode="json") for c in classified],
                "contradictions": [c.model_dump(mode="json") for c in contradictions],
                "retrieved_propositions": [p.model_dump(mode="json") for p in props],
                "confidence": confidence.model_dump(mode="json"),
            }
            
        except Exception as e:
            st.error(f" Query failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            return None
    
    else:
        # HTTP mode - call FastAPI
        try:
            resp = requests.post(
                f"{BACKEND_URL}/query",
                json={"query_text": query},
                timeout=60
            )
            if resp.ok:
                return resp.json()
            else:
                st.error(f" Query failed: {resp.text}")
                return None
        except requests.exceptions.ConnectionError:
            st.error(" Cannot connect to backend. Is FastAPI running on port 8000?")
            return None


def _render_results(result: dict) -> None:
    """
    Render the full structured answer - the WOW FACTOR of Arbiter.
    
    This renders:
    1. Prose answer at the top
    2. Individual claims with colored badges
    3. Confidence breakdown
    4. Sources list
    """
    
    # ---- Summary Stats Bar ----
    claims = result.get("claims", [])
    contradictions = result.get("contradictions", [])
    confidence = result.get("confidence", {})
    retrieved = result.get("retrieved_propositions", [])
    
    # Count claim types
    consensus_count = sum(1 for c in claims if c.get("status") == "consensus")
    disputed_count = sum(1 for c in claims if c.get("status") == "disputed")
    single_count = sum(1 for c in claims if c.get("status") == "single_source")
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Consensus", consensus_count)
    col2.metric("Disputed", disputed_count)
    col3.metric("Single Source", single_count)
    col4.metric("Confidence", f"{confidence.get('overall', 0)*100:.0f}%")
    
    st.divider()
    
    # ---- Prose Answer ----
    st.markdown('<div class="section-header">Answer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-box">{result.get("main_answer", "")}</div>',
        unsafe_allow_html=True
    )
    
    # ---- Individual Claims with Badges ----
    if claims:
        st.markdown('<div class="section-header">Claims Breakdown</div>', unsafe_allow_html=True)
        
        # Build proposition lookup for source display
        prop_lookup = {
            p.get("proposition_id", ""): p 
            for p in retrieved
        }
        
        for claim in claims:
            render_claim_card(claim, prop_lookup)
    
    # ---- Confidence Breakdown ----
    st.markdown('<div class="section-header">Confidence Breakdown</div>', unsafe_allow_html=True)
    
    conf_col1, conf_col2 = st.columns(2)
    
    with conf_col1:
        overall = confidence.get("overall", 0)
        st.markdown(f"**Overall Confidence: {overall*100:.0f}%**")
        st.progress(overall)
        
        ret_quality = confidence.get("retrieval_quality", 0)
        st.markdown(f"Retrieval Quality: {ret_quality*100:.0f}%")
        st.progress(ret_quality)
    
    with conf_col2:
        cons_ratio = confidence.get("consensus_ratio", 0)
        st.markdown(f"Consensus Ratio: {cons_ratio*100:.0f}%")
        st.progress(cons_ratio)
        
        src_coverage = confidence.get("source_coverage", 0)
        st.markdown(f"Source Coverage: {src_coverage*100:.0f}%")
        st.progress(src_coverage)
    
    if confidence.get("fallback_triggered"):
        st.warning(" Retrieval fallback was triggered - initial search found poor results.")
    
    # ---- Sources ----
    if retrieved:
        with st.expander(f" View {len(retrieved)} retrieved propositions"):
            for i, prop in enumerate(retrieved):
                st.markdown(f"**[{i+1}] {prop.get('doc_title', '')}** *(relevance: {prop.get('reranker_score', 0):.3f})*")
                st.markdown(f"> {prop.get('text', '')}")
                st.markdown(f"Section: `{prop.get('section_type', '')}` | ID: `{prop.get('proposition_id', '')[:20]}...`")
                if i < len(retrieved) - 1:
                    st.divider()


# ============================================================
# MAIN - Assemble the App
# ============================================================

def main():
    """Main entry point - renders sidebar + main area."""
    render_sidebar()
    render_main_area()


if __name__ == "__main__" or True:
    # Streamlit runs the whole script, so we call main() at module level
    main()
