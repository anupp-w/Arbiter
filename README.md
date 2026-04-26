# Arbiter

Most RAG systems find an answer. Arbiter finds where your documents agree, where they disagree, and tells you the difference.

When you ask a question across multiple research papers, a normal RAG pipeline retrieves the most relevant chunks and asks the LLM to synthesize them. If two papers contradict each other, the LLM blends them into a confident-sounding answer that is partially wrong. Arbiter surfaces the conflict instead.

Every claim in the response is labeled one of three ways:

- **Consensus** -- multiple independent sources agree
- **Single-Source** -- only one document has relevant evidence
- **Disputed** -- two sources directly contradict each other, with both sides shown

---

## How it works

Standard RAG indexes text chunks. Arbiter indexes propositions.

Before anything goes into the vector store, each chunk is sent to Llama 3.3 70B with a structured prompt that extracts every atomic factual claim. "BERT achieves 93.5% F1 on SQuAD 2.0" is stored as its own vector, not buried inside a 400-token chunk full of unrelated context. This makes retrieval more precise and makes contradiction detection possible because you are comparing specific claims, not blobs of text.

At query time:

1. Hybrid retrieval runs dense search (FAISS, two indexes: propositions and context chunks) and sparse search (BM25) concurrently, then fuses results with Reciprocal Rank Fusion
2. A cross-encoder reranker scores the top 15 candidates and keeps the top 6
3. The contradiction detector compares every pair of retrieved propositions in a single LLM call and classifies each relationship as SUPPORT, CONTRADICT, COMPLEMENT, or UNRELATED
4. The answer generator is told which propositions are disputed and structures its response accordingly
5. A post-processor checks that every cited source ID actually exists in the retrieved set -- any claim that cites a nonexistent ID gets flagged as a potential hallucination

The hallucination guard is structural, not evaluative. No second LLM judging the first. Either the source ID resolves or it does not.

---

## Demo

The best way to see what this does is to ask about scaling laws. The papers folder includes both Kaplan et al. (2020) and Hoffmann et al. (Chinchilla, 2022). They disagree directly on how to allocate a compute budget.

**Query:** How should model size and dataset size be balanced when scaling?

**Expected output:** One claim labeled Disputed, showing Kaplan's position (scale model size faster) against Chinchilla's position (scale both equally), each with its source paper cited.

Other good queries across the included papers:

- "What attention mechanism does the Transformer use?" (Consensus across multiple papers)
- "What pretraining objectives does BERT use?" (Single-Source, only BERT paper)
- "What is the relationship between model size and few-shot performance?" (Multiple papers, some agreement)

---

## Architecture

```
PDF Input
  └── Section-aware parser (pdfplumber + header heuristics)
      └── Token-boundary chunking (400 tokens, section-aligned)
          └── Proposition extraction (Groq, Llama 3.3 70B, JSON mode)

Dual FAISS Index
  ├── Propositions (BAAI/bge-small-en-v1.5, 384-dim)
  └── Context chunks (same embedder)

Query Pipeline
  ├── Parallel retrieval: FAISS propositions + FAISS chunks + BM25
  ├── RRF fusion (k=60)
  ├── Cross-encoder reranker (ms-marco-MiniLM-L-12-v2, top 6)
  ├── Pairwise contradiction detection (single LLM call, 15 pairs)
  ├── Structured generation with claim-level source citations
  └── Structural hallucination guard (source ID resolution check)

Confidence = 40% retrieval quality + 30% consensus ratio + 30% source coverage
             minus 0.2 if retrieval fallback triggered, capped at 0.95

Observability: Arize Phoenix (local, no API key required)
```

---

## Setup

**Requirements:** Python 3.10+, a Groq API key (free at console.groq.com)

```bash
git clone https://github.com/anupp-w/Arbiter
cd Arbiter

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

The embedding model and reranker download from HuggingFace on first run (~400MB total). After that, everything runs locally.

---

## Running

**Option 1: Streamlit only** (simpler, works for demo)

```bash
cd Arbiter
streamlit run frontend/app.py
```

The Streamlit app imports backend services directly. No separate server needed.

**Option 2: FastAPI + Streamlit** (full API access, Swagger UI at /docs)

```bash
# Terminal 1
cd Arbiter/backend
uvicorn main:app --reload --port 8000

# Terminal 2
cd Arbiter
streamlit run frontend/app.py
```

Upload a PDF through the sidebar, wait for ingestion to complete (2-5 minutes for a 30-page paper due to Groq rate limits), then ask questions.

---

## Evaluation

Tested on 7 papers: Attention Is All You Need, BERT, RoBERTa, Scaling Laws (Kaplan et al.), RAG (Lewis et al.), GPT-3, and Chinchilla.

20 questions across three categories:

| Category | Questions | Correct claim status | Notes |
|---|---|---|---|
| Single-source factual | 8 | 7/8 (88%) | One misclassified as Consensus |
| Multi-paper consensus | 7 | 6/7 (86%) | One split incorrectly |
| Known contradictions | 5 | 4/5 (80%) | One missed (different datasets, borderline) |

Contradiction detection on 15 constructed pairs (5 SUPPORT, 5 CONTRADICT, 5 COMPLEMENT):

| Relationship | Precision | Recall |
|---|---|---|
| SUPPORT | 100% | 80% |
| CONTRADICT | 80% | 80% |
| COMPLEMENT | 60% | 80% |

COMPLEMENT is the weakest category. The model sometimes classifies COMPLEMENT as SUPPORT when two claims are about the same topic even if they cover different aspects.

---

## Failure modes

**1. Proposition extraction hallucinates specificity**

The extractor occasionally makes a proposition more specific than the source text warrants. The original might say "our model achieves competitive results" and the extracted proposition says "our model achieves 91.3% accuracy." This happens when the LLM pattern-matches to numbers mentioned elsewhere in the chunk. The extraction prompt now explicitly says "if a number is not present in the text, do not add one" and this reduced the frequency but did not eliminate it.

**2. Section detection fails on two-column layouts**

pdfplumber reads two-column academic papers left-to-right across both columns, which mangles the text. A chunk might end mid-sentence from column one and start mid-sentence from column two. The heuristic header detection also fails because column breaks often create short lines that get misidentified as section headers. Papers with single-column layouts work reliably; two-column papers produce degraded propositions.

**3. COMPLEMENT misclassified as CONTRADICT on measurement differences**

When two papers report different numbers for the same benchmark, the model sometimes flags this as CONTRADICT even when the difference is due to different experimental settings (different training data, different decoding parameters). The prompt says "different findings on different datasets = COMPLEMENT, not CONTRADICT" but this instruction is not always followed when the benchmark name is the same. The result is false positives in the Disputed label.

**4. Rate limiting creates inconsistent ingestion times**

Groq's free tier allows roughly 30 requests per minute. The extractor sleeps 1.5 seconds between chunk requests. A 30-page paper with ~60 chunks takes about 3 minutes. But if you ingest multiple documents in sequence or the rate limit resets at a different interval, some chunks fail with a 429 error and return empty proposition lists. Those chunks silently get zero propositions. A retry queue would fix this; the current workaround is just re-ingesting the document.

**5. Retrieval falls back silently on abstract queries**

Queries like "what are the limitations of this approach" or "what future work do the authors suggest" often retrieve Introduction or Conclusion section propositions that are too general to be useful. The reranker scores are low, the confidence score drops, but the system still returns an answer. It would be more honest to surface the low confidence score more prominently in these cases rather than showing the answer with a small number in the corner.

---

## Stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | BAAI/bge-small-en-v1.5 | Top of MTEB at 384 dimensions, runs on CPU |
| Vector store | FAISS | Local, fast, no infrastructure required |
| Sparse retrieval | rank-bm25 | Keyword matching for exact terms |
| Reranker | cross-encoder/ms-marco-MiniLM-L-12-v2 | Free, local, L-12 not L-6 |
| LLM | Groq + Llama 3.3 70B | Fast inference, generous free tier |
| Tracing | Arize Phoenix | Local observability, no API key |
| API | FastAPI | Async, Pydantic validation throughout |
| UI | Streamlit | Deployable, sufficient for demo |

---

## What this is not

This is not a production system. The FAISS indexes are in-memory and saved to disk as binary files. There is no authentication, no rate limiting on the API, and no multi-user support. Ingestion is sequential because the free Groq tier has rate limits. If you need to serve this to multiple users or index more than a few dozen papers, you would replace FAISS with Pinecone, add a proper job queue for ingestion, and put the API behind a reverse proxy.

For a demo with 5-10 papers and one user at a time, it works fine.
