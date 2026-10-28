"""
BMU Retrieval Microservice
==========================
Standalone FastAPI microservice providing:
- Local SentenceTransformer embedding inference
- ChromaDB Cloud vector similarity search
- Hybrid BM25-style keyword + dense semantic ranking
- Health check and monitoring endpoint
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from voyage_client import voyage_client

# Load environment variables
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# Collections
BGE_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "bmu_bge_knowledge")
PROD_COLLECTION_NAME = "bmu_summaries"
VOYAGE_COLLECTION_NAME = os.getenv("VOYAGE_COLLECTION_NAME", "bmu_voyage_knowledge")
DEFAULT_RETRIEVAL_ENGINE = os.getenv("RETRIEVAL_ENGINE", "bge").lower()

# Embedding Models
BGE_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
EMBEDDING_MODEL_NAME = BGE_MODEL_NAME
MINILM_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BMU Semantic Retrieval Microservice",
    description="Dedicated microservice for BGE (1,024-d), MiniLM, and Voyage AI embeddings, ChromaDB search, and hybrid ranking.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy Loaded Resources
# ---------------------------------------------------------------------------
_chroma_client = None
_bge_collection = None
_bge_embedder = None
_prod_collection = None
_voyage_collection = None
_minilm_embedder = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
            raise RuntimeError("Missing required ChromaDB configuration in environment (.env).")
        _chroma_client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
    return _chroma_client


def get_bge_collection():
    global _bge_collection
    if _bge_collection is None:
        client = get_chroma_client()
        _bge_collection = client.get_or_create_collection(name=BGE_COLLECTION_NAME)
    return _bge_collection


def get_bge_embedder():
    global _bge_embedder
    if _bge_embedder is None:
        _bge_embedder = SentenceTransformer(BGE_MODEL_NAME)
    return _bge_embedder


def get_prod_collection():
    global _prod_collection
    if _prod_collection is None:
        client = get_chroma_client()
        _prod_collection = client.get_or_create_collection(name=PROD_COLLECTION_NAME)
    return _prod_collection


def get_voyage_collection():
    global _voyage_collection
    if _voyage_collection is None:
        client = get_chroma_client()
        _voyage_collection = client.get_or_create_collection(name=VOYAGE_COLLECTION_NAME)
    return _voyage_collection


def get_embedder():
    """Backward compatibility helper."""
    return get_bge_embedder()


def get_minilm_embedder():
    global _minilm_embedder
    if _minilm_embedder is None:
        _minilm_embedder = SentenceTransformer(MINILM_MODEL_NAME)
    return _minilm_embedder


def get_chroma_and_embedder():
    """Backward compatibility helper."""
    return get_bge_collection(), get_bge_embedder()


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=8, ge=1, le=50, description="Maximum number of ranked results to return")
    engine: Optional[str] = Field(
        default=None,
        description="'bge' (1,024-d bmu_bge_knowledge), 'minilm' (384-d bmu_summaries), or 'voyage' (1,024-d bmu_voyage_knowledge). Defaults to env RETRIEVAL_ENGINE.",
    )


class RankedDocument(BaseModel):
    summary: str
    url: str
    score: float
    semantic_sim: float
    keyword_score: float
    title: Optional[str] = ""
    engine: Optional[str] = "bge"


class RetrieveResponse(BaseModel):
    query: str
    count: int
    latency_ms: float
    engine: str
    collection: str
    results: List[RankedDocument]


class HealthResponse(BaseModel):
    status: str
    service: str
    default_engine: str
    bge_model: str
    bge_collection: str
    bge_dimension: int = 1024
    bge_document_count: Optional[int] = None
    minilm_model: str
    minilm_collection: str
    minilm_document_count: Optional[int] = None
    voyage_model: str
    voyage_collection: str
    voyage_document_count: Optional[int] = None
    voyage_configured: bool
    latency_ms: float
    # Backward compatibility fields
    model: str
    collection: str
    document_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Ranking Logic
# ---------------------------------------------------------------------------
def _keyword_score(query: str, text: str) -> float:
    """Calculates word overlap ratio between query and candidate text."""
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    if not query_words:
        return 0.0
    overlap = query_words.intersection(text_words)
    return len(overlap) / len(query_words)


# ---------------------------------------------------------------------------
# Microservice Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    start_t = time.perf_counter()
    bge_count = None
    prod_count = None
    voyage_count = None
    status_str = "ok"

    try:
        bge_col = get_bge_collection()
        bge_count = bge_col.count()
    except Exception as exc:
        status_str = f"bge_degraded: {exc}"

    try:
        prod_col = get_prod_collection()
        prod_count = prod_col.count()
    except Exception as exc:
        pass

    try:
        voyage_col = get_voyage_collection()
        voyage_count = voyage_col.count()
    except Exception as exc:
        pass

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    return HealthResponse(
        status=status_str,
        service="BMU Retrieval Microservice",
        default_engine=DEFAULT_RETRIEVAL_ENGINE,
        bge_model=BGE_MODEL_NAME,
        bge_collection=BGE_COLLECTION_NAME,
        bge_dimension=1024,
        bge_document_count=bge_count,
        minilm_model=MINILM_MODEL_NAME,
        minilm_collection=PROD_COLLECTION_NAME,
        minilm_document_count=prod_count,
        voyage_model=voyage_client.model_name,
        voyage_collection=VOYAGE_COLLECTION_NAME,
        voyage_document_count=voyage_count,
        voyage_configured=voyage_client.is_configured,
        latency_ms=latency_ms,
        # Backward compatibility
        model=BGE_MODEL_NAME,
        collection=BGE_COLLECTION_NAME,
        document_count=bge_count,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    start_t = time.perf_counter()
    clean_query = req.query.strip().lower()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty or whitespace.",
        )

    engine = (req.engine or DEFAULT_RETRIEVAL_ENGINE).lower()
    if engine not in ["bge", "minilm", "voyage"]:
        engine = "bge"

    try:
        ranked: List[RankedDocument] = []
        n_fetch = min(max(req.top_k, 8), 25)

        if engine == "bge":
            # =========================================================
            # BGE 1,024-dim Engine (BAAI/bge-large-en-v1.5, bmu_bge_knowledge)
            # =========================================================
            col = get_bge_collection()
            embedder = get_bge_embedder()
            # BGE retrieval prompt instruction for highest ranking quality
            q_text = f"Represent this sentence for searching relevant passages: {clean_query}"
            query_emb = embedder.encode([q_text], normalize_embeddings=True).tolist()
            raw_results = col.query(query_embeddings=query_emb, n_results=n_fetch)
            active_collection = BGE_COLLECTION_NAME

            documents = raw_results.get("documents", [[]])[0]
            metadatas = raw_results.get("metadatas", [[]])[0]
            distances = raw_results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                sem_sim = max(0.0, 1.0 - dist)
                kw_sc = _keyword_score(clean_query, doc)
                final_sc = (0.70 * sem_sim) + (0.30 * kw_sc)

                ranked.append(
                    RankedDocument(
                        summary=doc,
                        url=meta.get("url", ""),
                        title=meta.get("title", ""),
                        score=round(float(final_sc), 4),
                        semantic_sim=round(float(sem_sim), 4),
                        keyword_score=round(float(kw_sc), 4),
                        engine="bge",
                    )
                )

        elif engine == "voyage":
            # =========================================================
            # Voyage AI Engine (1024-dim, bmu_voyage_knowledge)
            # =========================================================
            if not voyage_client.is_configured:
                raise RuntimeError("Voyage AI client is not configured in .env.")
            col = get_voyage_collection()
            query_emb = [voyage_client.embed_query(clean_query)]
            raw_results = col.query(query_embeddings=query_emb, n_results=n_fetch)
            active_collection = VOYAGE_COLLECTION_NAME

            documents = raw_results.get("documents", [[]])[0]
            metadatas = raw_results.get("metadatas", [[]])[0]
            distances = raw_results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                sem_sim = max(0.0, 1.0 - dist)
                kw_sc = _keyword_score(clean_query, doc)
                final_sc = (0.68 * sem_sim) + (0.32 * kw_sc)

                ranked.append(
                    RankedDocument(
                        summary=doc,
                        url=meta.get("url", ""),
                        title=meta.get("title", ""),
                        score=round(float(final_sc), 4),
                        semantic_sim=round(float(sem_sim), 4),
                        keyword_score=round(float(kw_sc), 4),
                        engine="voyage",
                    )
                )

        else:
            # =========================================================
            # MiniLM Legacy Engine (384-dim, bmu_summaries)
            # =========================================================
            col = get_prod_collection()
            embedder = get_minilm_embedder()
            query_emb = embedder.encode([clean_query]).tolist()
            raw_results = col.query(query_embeddings=query_emb, n_results=n_fetch)
            active_collection = PROD_COLLECTION_NAME

            documents = raw_results.get("documents", [[]])[0]
            metadatas = raw_results.get("metadatas", [[]])[0]
            distances = raw_results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                sem_sim = max(0.0, 1.0 - dist)
                kw_sc = _keyword_score(clean_query, doc)
                final_sc = (0.68 * sem_sim) + (0.32 * kw_sc)

                ranked.append(
                    RankedDocument(
                        summary=doc,
                        url=meta.get("url", ""),
                        title=meta.get("title", ""),
                        score=round(float(final_sc), 4),
                        semantic_sim=round(float(sem_sim), 4),
                        keyword_score=round(float(kw_sc), 4),
                        engine="minilm",
                    )
                )

        ranked.sort(key=lambda x: x.score, reverse=True)
        final_results = ranked[:req.top_k]

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval query execution failed ({engine}): {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    return RetrieveResponse(
        query=req.query,
        count=len(final_results),
        latency_ms=latency_ms,
        engine=engine,
        collection=active_collection,
        results=final_results,
    )


@app.post("/retrieve/compare")
def retrieve_compare(req: RetrieveRequest) -> Dict[str, Any]:
    """
    Model Comparison Tool: Runs query against BOTH BGE 1,024-d (bmu_bge_knowledge)
    and MiniLM 384-d (bmu_summaries) side-by-side to demonstrate accuracy and score improvements.
    """
    clean_query = req.query.strip().lower()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_bge = time.perf_counter()
    bge_res = retrieve(RetrieveRequest(query=clean_query, top_k=req.top_k, engine="bge"))
    bge_time = round((time.perf_counter() - start_bge) * 1000, 2)

    start_minilm = time.perf_counter()
    minilm_res = retrieve(RetrieveRequest(query=clean_query, top_k=req.top_k, engine="minilm"))
    minilm_time = round((time.perf_counter() - start_minilm) * 1000, 2)

    return {
        "query": req.query,
        "bge_1024_primary": {
            "collection": BGE_COLLECTION_NAME,
            "model": BGE_MODEL_NAME,
            "dimension": 1024,
            "latency_ms": bge_time,
            "count": bge_res.count,
            "top_url": bge_res.results[0].url if bge_res.results else None,
            "top_score": bge_res.results[0].score if bge_res.results else 0.0,
            "results": bge_res.results[:3],
        },
        "minilm_384_legacy": {
            "collection": PROD_COLLECTION_NAME,
            "model": MINILM_MODEL_NAME,
            "dimension": 384,
            "latency_ms": minilm_time,
            "count": minilm_res.count,
            "top_url": minilm_res.results[0].url if minilm_res.results else None,
            "top_score": minilm_res.results[0].score if minilm_res.results else 0.0,
            "results": minilm_res.results[:3],
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("RETRIEVAL_HOST", "0.0.0.0")
    port = int(os.getenv("RETRIEVAL_SERVICE_PORT", "8001"))
    print(f"Starting BMU Retrieval Microservice on http://{host}:{port}...")
    uvicorn.run("retrieval_service:app", host=host, port=port, reload=False)

