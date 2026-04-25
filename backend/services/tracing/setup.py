# ============================================================
# services/tracing/setup.py - Arize Phoenix Observability
# ============================================================
#
# WHAT IS TRACING?
# ----------------
# When a user asks a query, a lot happens under the hood:
# 1. Hybrid search (FAISS + BM25)
# 2. Reranking (Cross-Encoder)
# 3. LLM call for contradiction detection
# 4. LLM call for generation
#
# If a query takes 10 seconds or returns a bad result, you need 
# to know exactly which step was slow or produced bad data.
# Tracing records the inputs, outputs, and duration of every step.
#
# PHOENIX BY ARIZE:
# -----------------
# We use Phoenix instead of LangSmith because Phoenix runs 
# 100% locally and is completely free. No API keys needed.
# It provides a beautiful UI at http://localhost:6006 to inspect 
# your RAG traces.
#
# OpenTelemetry (OTel):
# ---------------------
# OTel is the industry standard for tracing. Phoenix acts as an 
# OTel collector. We configure our app to send OTel spans to Phoenix.
# ============================================================

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from fastapi import FastAPI
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings


def setup_tracing(app: FastAPI) -> None:
    """
    Configure OpenTelemetry to send traces to the local Phoenix server.
    
    Call this once on application startup.
    """
    print(" Initializing Phoenix OpenTelemetry tracing...")
    
    # Create the tracer provider
    provider = TracerProvider()
    
    # Configure the exporter to send traces to the local Phoenix server
    # Phoenix exposes an OTLP HTTP endpoint on port 6006 by default
    endpoint = f"http://localhost:6006/v1/traces"
    
    try:
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)
        
        # Set this as the global tracer provider
        trace.set_tracer_provider(provider)
        
        print(f" Tracing configured. Phoenix UI available at http://localhost:6006")
        print("   (Note: You need to start the Phoenix server separately: python -m phoenix.server.main)")
        
    except Exception as e:
        print(f"  Could not configure tracing: {e}")


def get_tracer(name: str):
    """
    Get a tracer instance for manual span creation.
    
    Example:
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("my_operation"):
            do_something()
    """
    return trace.get_tracer(name)
