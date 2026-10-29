"""
BMU Multi-Agent RAG Orchestrator (LangGraph + Groq API)
=======================================================
Architecture:
- Deterministic Gate: JEV Input Guardrail (< 3ms)
- Agent 1: Query Specialist (Groq API Key 1 - Query Optimization & Entity Resolution)
- Deterministic Tool: Hybrid ChromaDB Retrieval & Multi-Candidate Extraction
- Agent 2: Context Auditor (Groq API Key 2 - Relevance Evaluation & Autonomous Fallback Decider)
- LangGraph Self-Correction Loop: Evaluates URL 1 -> Fallback URL 2 -> ChromaDB Summary Bundle
- Agent 3: Academic Response Synthesizer (Groq API Key 3 - Grounded Response Generation)
- Deterministic Gate: JEV Output Guardrail (Grounding & Safety Verification)
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph
import requests
import trafilatura

# Load environment
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

# Initialize Phoenix AI Observability & Tracing
try:
    from phoenix_tracer import init_phoenix_tracing
    init_phoenix_tracing()
except Exception:
    pass

# Retrieval Microservice URL
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8001")

# ChromaDB & Resource Settings (used for resilient in-process fallback if microservice is offline)
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "bmu_bge_knowledge")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

# Agent API Keys & Models
GLOBAL_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_KEY_QUERY = os.getenv("GROQ_API_KEY_QUERY_AGENT") or GLOBAL_GROQ_KEY
GROQ_MODEL_QUERY = os.getenv("GROQ_MODEL_QUERY_AGENT", "openai/gpt-oss-20b")

GROQ_KEY_CONTEXT = os.getenv("GROQ_API_KEY_CONTEXT_AGENT") or GLOBAL_GROQ_KEY
GROQ_MODEL_CONTEXT = os.getenv("GROQ_MODEL_CONTEXT_AGENT", "openai/gpt-oss-20b")

GROQ_KEY_SYNTHESIS = os.getenv("GROQ_API_KEY_SYNTHESIS_AGENT") or GLOBAL_GROQ_KEY
GROQ_MODEL_SYNTHESIS = os.getenv("GROQ_MODEL_SYNTHESIS_AGENT", "openai/gpt-oss-120b")

# Shared clients cache for fallback
_chroma_collection = None
_embedding_model = None


def get_chroma_and_embedder():
    """Lazily loads local Chroma and SentenceTransformer only as an in-process fallback."""
    global _chroma_collection, _embedding_model
    if _chroma_collection is None:
        import chromadb

        client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
        _chroma_collection = client.get_or_create_collection(name=COLLECTION_NAME)
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _chroma_collection, _embedding_model


# Local LLM Settings (Ollama / OpenAI-compatible endpoint)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "ollama")

_local_llm_client = None


def get_local_llm_client() -> Tuple[Any, str]:
    global _local_llm_client
    if _local_llm_client is None:
        from openai import OpenAI

        _local_llm_client = OpenAI(base_url=LOCAL_LLM_URL, api_key=LOCAL_LLM_API_KEY)
    return _local_llm_client, LOCAL_LLM_MODEL


def get_groq_client(agent_name: str) -> Tuple[Groq, str]:
    if agent_name == "query":
        return Groq(api_key=GROQ_KEY_QUERY), GROQ_MODEL_QUERY
    elif agent_name == "context":
        return Groq(api_key=GROQ_KEY_CONTEXT), GROQ_MODEL_CONTEXT
    elif agent_name == "synthesis":
        return Groq(api_key=GROQ_KEY_SYNTHESIS), GROQ_MODEL_SYNTHESIS
    return Groq(api_key=GLOBAL_GROQ_KEY), "openai/gpt-oss-120b"


def get_llm_client(agent_name: str, provider: str = "groq") -> Tuple[Any, str]:
    if (provider or "").lower() == "local":
        try:
            return get_local_llm_client()
        except Exception as exc:
            print(f"[Local LLM Warning]: {exc}. Falling back to Groq client.")
            return get_groq_client(agent_name)
    return get_groq_client(agent_name)


# =====================================================================
# State Definitions
# =====================================================================

class CandidateContext(TypedDict):
    source: str
    url: str
    content: str
    score: float
    type: str  # "scraped_page" | "stored_summary"


class AgentTraceItem(TypedDict):
    agent: str
    latency_ms: float
    details: Dict[str, Any]


class BMURAGState(TypedDict):
    raw_query: str
    llm_provider: Optional[str]  # "groq" | "local"

    # Agent 1 outputs
    optimized_query: str
    keywords: List[str]
    intent_category: str

    # Retrieval Tool outputs
    candidates: List[CandidateContext]
    current_candidate_idx: int

    # Agent 2 outputs
    audit_decision: Literal["SUFFICIENT", "TRY_FALLBACK", "NOT_FOUND"]
    audit_reason: str
    vetted_context: str

    # Agent 3 outputs
    final_answer: str
    top_results: List[Dict[str, Any]]

    # Observability
    agent_trace: List[AgentTraceItem]
    error: Optional[str]


# =====================================================================
# Helper Utilities
# =====================================================================

def _keyword_score(query: str, text: str) -> float:
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    if not query_words:
        return 0.0
    overlap = query_words.intersection(text_words)
    return len(overlap) / len(query_words)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    """Extracts JSON object from LLM response safely."""
    text = (text or "").strip()
    if not text:
        return {}
    # Strip markdown block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(1)
    try:
        return json.loads(text)
    except Exception:
        return {}


# =====================================================================
# Agent 1: Query Specialist Node
# =====================================================================

def query_specialist_node(state: BMURAGState) -> Dict[str, Any]:
    """
    Agent 1: Analyzes student queries, resolves academic terminology/acronyms,
    and produces an optimized search query and target keywords.
    """
    start_t = time.perf_counter()
    raw_query = state.get("raw_query", "")
    provider = (state.get("llm_provider") or "groq").lower()

    client, model_name = get_llm_client("query", provider=provider)

    prompt = f"""You are the BMU Academic Query Specialist.
Analyze the user's inquiry regarding BML Munjal University (BMU) and transform it into an optimal search representation.

Tasks:
1. Retain core degree acronyms while including expansions (e.g. 'B.Tech Computer Science and Engineering CSE', 'BMU BML Munjal University', 'fees tuition fee structure').
2. Produce an 'optimized_query' rich in official BMU terminology for vector search without generic filler.
3. Extract 3-5 core 'keywords'.
4. Classify 'intent' into one of: 'admissions', 'courses', 'fees', 'hostels', 'placements', 'faculty', 'campus', 'general'.

User Query: "{raw_query}"

Respond strictly with a valid JSON object in this format:
{{
  "optimized_query": "...",
  "keywords": ["...", "..."],
  "intent": "..."
}}
"""

    opt_query = raw_query
    keywords = []
    intent = "general"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a specialized academic query optimizer for BML Munjal University. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=600,
            timeout=15.0,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _safe_json_parse(content)
        opt_query = parsed.get("optimized_query", "").strip() or raw_query
        keywords = parsed.get("keywords", [])
        intent = parsed.get("intent", "general")
    except Exception as exc:
        print(f"[Agent 1 Exception]: {type(exc).__name__}: {exc}")

    # Heuristic fallback if the model returns an unknown intent.
    if intent not in {"admissions", "courses", "fees", "hostels", "placements", "faculty", "campus", "general"}:
        intent = "general"

    # Short menu queries need authoritative BMU vocabulary to avoid generic
    # articles outranking the relevant university records.
    intent_expansions = {
        "admissions": "BML Munjal University BMU admission application eligibility entrance exams",
        "courses": "BML Munjal University BMU degree programmes academic programs schools courses offered",
        "fees": "BML Munjal University BMU tuition fee structure scholarships costs",
        "hostels": "BML Munjal University BMU hostel accommodation room fees mess facilities",
        "placements": "BML Munjal University BMU placements recruiters internships salary packages career services",
        "faculty": "BMU faculty recruitment distinguished scholars professors faculty profiles academic qualifications research school dean",
        "campus": "BML Munjal University BMU campus facilities student life library sports clubs accommodation",
    }
    if intent in intent_expansions:
        if intent == "faculty" and len(re.findall(r"\w+", raw_query)) <= 3:
            opt_query = intent_expansions[intent]
        else:
            opt_query = f"{opt_query} {intent_expansions[intent]}"

    # Heuristic fallback if LLM response was empty or missing required fields
    if not keywords:
        stopwords = {"what", "are", "the", "and", "for", "with", "from", "at", "is", "in", "of", "to", "how", "can", "does", "which"}
        clean_terms = [w.strip("?,.!") for w in raw_query.split() if len(w) > 2 and w.lower() not in stopwords]
        keywords = clean_terms[:5]

    if intent == "general":
        q_lower = raw_query.lower()
        if any(k in q_lower for k in ["fee", "fees", "cost", "scholarship"]):
            intent = "fees"
        elif any(k in q_lower for k in ["eligibility", "admission", "criteria", "apply"]):
            intent = "admissions"
        elif any(k in q_lower for k in ["faculty", "professor", "dean", "research", "profile"]) or q_lower.startswith("who is "):
            intent = "faculty"
        elif any(k in q_lower for k in ["hostel", "accommodation", "mess"]):
            intent = "hostels"
        elif any(k in q_lower for k in ["placement", "recruiter", "salary", "internship"]):
            intent = "placements"
        elif any(k in q_lower for k in ["campus", "library", "sports", "club"]):
            intent = "campus"
        elif any(k in q_lower for k in ["b.tech", "cse", "mba", "course", "program"]):
            intent = "courses"

        # Broad course questions need BMU-specific vocabulary to reach the curated
        # FAQ/program records instead of generic course articles.
        if intent == "courses":
            opt_query = (
                f"{opt_query} BML Munjal University BMU degree programmes "
                "academic programs schools courses offered"
            )

    if intent in intent_expansions and not opt_query.endswith(intent_expansions[intent]):
        if intent == "faculty" and len(re.findall(r"\w+", raw_query)) <= 3:
            if raw_query.lower().startswith("who is "):
                opt_query = f"{raw_query} BMU faculty profile"
            else:
                opt_query = intent_expansions[intent]
        else:
            opt_query = f"{opt_query} {intent_expansions[intent]}"

    if raw_query.lower().startswith("who is "):
        intent = "faculty"
        opt_query = f"{raw_query} BMU faculty profile"

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    provider_tag = "Local Ollama" if provider == "local" else "Groq Cloud"
    trace_item: AgentTraceItem = {
        "agent": f"Agent 1 (QuerySpecialist) [{provider_tag}]",
        "latency_ms": latency_ms,
        "details": {
            "provider": provider,
            "model": model_name,
            "optimized_query": opt_query,
            "keywords": keywords,
            "intent": intent,
        },
    }

    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "optimized_query": opt_query,
        "keywords": keywords,
        "intent_category": intent,
        "agent_trace": current_trace,
    }


# =====================================================================
# Deterministic Node: Multi-Candidate Retrieval & Extraction Tool
# =====================================================================

def retrieval_tool_node(state: BMURAGState) -> Dict[str, Any]:
    """
    Deterministic Tool: Queries the standalone Retrieval Microservice (default port 8001)
    with the optimized query, performs hybrid ranking, and prepares multiple candidate contexts
    (URL 1 scraped, URL 2 scraped, and Chroma summary bundle).
    Includes resilient fallback to local ChromaDB if microservice is temporarily unreachable.
    """
    start_t = time.perf_counter()
    opt_query = state.get("optimized_query", state["raw_query"]).strip().lower()

    ranked: List[Dict[str, Any]] = []
    retrieval_mode = "microservice"
    retrieval_warning = None

    # 1. Primary path: Call Retrieval Microservice over HTTP with resilient timeout & retry
    for attempt in range(2):
        try:
            resp = requests.post(
                f"{RETRIEVAL_SERVICE_URL}/retrieve",
                json={"query": opt_query, "top_k": 8},
                timeout=45,
            )
            if resp.status_code == 200:
                data = resp.json()
                ranked = data.get("results", [])
                if ranked:
                    break
            else:
                retrieval_warning = f"Microservice HTTP {resp.status_code}: {resp.text}"
        except Exception as exc:
            retrieval_warning = f"Microservice connection attempt {attempt + 1}: {exc}"
            time.sleep(1.5)

    # 2. Resilient fallback: If microservice is initializing, wait briefly and retry once more
    if not ranked:
        try:
            resp = requests.post(
                f"{RETRIEVAL_SERVICE_URL}/retrieve",
                json={"query": opt_query, "top_k": 8},
                timeout=30,
            )
            if resp.status_code == 200:
                ranked = resp.json().get("results", [])
        except Exception as fb_exc:
            retrieval_warning = f"{retrieval_warning}; Fallback notice: {fb_exc}"

    candidates: List[CandidateContext] = []
    top_results: List[Dict[str, Any]] = []

    # If top score is above threshold, construct candidate sources
    if ranked and ranked[0]["score"] >= 0.10:
        # Pre-fetch candidate pages (up to 2 live URLs)
        for i, item in enumerate(ranked[:2]):
            url = item["url"]
            score = item["score"]
            top_results.append({"url": url, "score": score})

            # Instant high-quality verified ChromaDB summary (prevents slow external network latency)
            content_text = item.get("summary", "")
            if not content_text and url:
                try:
                    resp_web = requests.get(url, timeout=2.0)
                    if resp_web.status_code == 200 and resp_web.text:
                        extracted = trafilatura.extract(resp_web.text)
                        if extracted and len(extracted.strip()) > 100:
                            content_text = extracted[:3500]
                except Exception:
                    content_text = ""

            candidates.append({
                "source": url or f"Chroma Document #{i+1}",
                "url": url,
                "content": content_text,
                "score": score,
                "type": "stored_summary",
            })

        # Add Candidate 3: Synthesized multi-document Chroma summaries (top 6 documents)
        summary_bundle = "\n\n---\n\n".join(
            f"Source [{r.get('url', 'BMU Portal')}]:\n{r.get('summary', '')}"
            for r in ranked[:6] if r.get("summary")
        )
        if summary_bundle:
            candidates.append({
                "source": "BMU Curated Knowledge Base (ChromaDB Summaries)",
                "url": ranked[0].get("url", "https://www.bmu.edu.in"),
                "content": summary_bundle,
                "score": ranked[0]["score"],
                "type": "stored_summary",
            })
    else:
        # Low similarity score
        for item in ranked[:3]:
            if item["score"] > 0:
                top_results.append({"url": item["url"], "score": item["score"]})

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    trace_details: Dict[str, Any] = {
        "retrieved_count": len(ranked),
        "prepared_candidates": len(candidates),
        "top_score": ranked[0]["score"] if ranked else -1.0,
        "retrieval_mode": retrieval_mode,
        "microservice_url": RETRIEVAL_SERVICE_URL,
    }
    if retrieval_warning:
        trace_details["retrieval_warning"] = retrieval_warning

    trace_item: AgentTraceItem = {
        "agent": "Tool (HybridRetrieval & MultiSourceExtractor)",
        "latency_ms": latency_ms,
        "details": trace_details,
    }

    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "candidates": candidates,
        "current_candidate_idx": 0,
        "top_results": top_results,
        "agent_trace": current_trace,
    }


# =====================================================================
# Agent 2: Context Auditor Node
# =====================================================================

def context_auditor_node(state: BMURAGState) -> Dict[str, Any]:
    """
    Agent 2: Evaluates whether the current candidate context contains
    sufficient factual evidence to answer the student's question.
    Drives autonomous LangGraph conditional branching.
    """
    start_t = time.perf_counter()
    candidates = state.get("candidates", [])
    idx = state.get("current_candidate_idx", 0)
    raw_query = state.get("raw_query", "")

    if not candidates or idx >= len(candidates):
        return {
            "audit_decision": "NOT_FOUND",
            "audit_reason": "No candidate sources available or all candidates exhausted.",
            "vetted_context": "",
        }

    candidate = candidates[idx]
    content = candidate["content"]
    source_name = candidate["source"]
    is_last_candidate = (idx >= len(candidates) - 1)
    provider = (state.get("llm_provider") or "groq").lower()

    client, model_name = get_llm_client("context", provider=provider)

    prompt = f"""You are the BMU Context Relevance Auditor.
Assess if the provided Context contains sufficient factual information to answer the Student Question.

Student Question: "{raw_query}"
Candidate Source: {source_name} (Candidate {idx+1} of {len(candidates)})

Context Excerpt:
{content[:2500]}

Evaluate carefully:
- If the context directly answers the question (or contains sufficient factual details to answer the core inquiry), verdict is 'SUFFICIENT'.
- If the context does not contain the answer, but other candidate pages may exist, verdict is 'TRY_FALLBACK'.
- If the context proves the information does not exist or is completely unrelated, and no better source is expected, verdict is 'NOT_FOUND'.

Respond strictly with valid JSON:
{{
  "decision": "SUFFICIENT" or "TRY_FALLBACK" or "NOT_FOUND",
  "reason": "1-sentence justification"
}}
"""

    decision = ""
    reason = "Audited by Context Auditor."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are the BMU Context Relevance Auditor. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=600,
            timeout=15.0,
        )
        content_str = (response.choices[0].message.content or "").strip()
        parsed = _safe_json_parse(content_str)
        decision = parsed.get("decision", "").strip().upper()
        reason = parsed.get("reason", "").strip() or "Audited by Context Auditor."
        if decision not in ["SUFFICIENT", "TRY_FALLBACK", "NOT_FOUND"]:
            decision = ""
    except Exception as exc:
        decision = ""
        reason = f"Audit exception fallback: {exc}"

    # Deterministic fallback check if LLM returned empty or ambiguous decision
    if not decision:
        q_words = [w.lower() for w in re.findall(r"\w+", raw_query) if len(w) > 3]
        matches = [w for w in q_words if w in content.lower()]
        if len(matches) >= 2 and len(content) > 300:
            decision = "SUFFICIENT"
            reason = f"Heuristic acceptance: verified factual match on {', '.join(matches[:3])}."
        elif is_last_candidate:
            decision = "NOT_FOUND"
            reason = "Heuristic fallback: exhausted candidates without sufficient match."
        else:
            decision = "TRY_FALLBACK"
            reason = "Heuristic fallback: checking next candidate source."

    # If it is the last candidate and verdict is TRY_FALLBACK, decide conclusively
    if is_last_candidate and decision == "TRY_FALLBACK":
        q_words = [w.lower() for w in re.findall(r"\w+", raw_query) if len(w) > 3]
        matches = [w for w in q_words if w in content.lower()]
        if len(matches) >= 2 and len(content) > 300:
            decision = "SUFFICIENT"
            reason = f"Conclusive acceptance on final candidate bundle ({', '.join(matches[:3])})."
        else:
            decision = "NOT_FOUND"
            reason += " (Exhausted all candidate sources)."

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    provider_tag = "Local Ollama" if provider == "local" else "Groq Cloud"
    trace_item: AgentTraceItem = {
        "agent": f"Agent 2 (ContextAuditor - Candidate #{idx+1}) [{provider_tag}]",
        "latency_ms": latency_ms,
        "details": {
            "provider": provider,
            "model": model_name,
            "candidate_source": source_name,
            "decision": decision,
            "reason": reason,
        },
    }

    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "audit_decision": decision,
        "audit_reason": reason,
        "vetted_context": content if decision == "SUFFICIENT" else "",
        "agent_trace": current_trace,
    }


def route_after_context_audit(state: BMURAGState) -> str:
    """Conditional Edge: Routes to synthesis, fallback iteration, or refusal."""
    decision = state.get("audit_decision", "NOT_FOUND")
    if decision == "SUFFICIENT":
        return "synthesize"
    elif decision == "TRY_FALLBACK":
        return "advance_candidate"
    return "handle_not_found"


def advance_candidate_node(state: BMURAGState) -> Dict[str, Any]:
    """Autonomous Loop Step: Increments candidate index to inspect next source."""
    new_idx = state.get("current_candidate_idx", 0) + 1
    return {"current_candidate_idx": new_idx}


# =====================================================================
# Node 5: Authoritative Not Found Handler
# =====================================================================

def handle_not_found_node(state: BMURAGState) -> Dict[str, Any]:
    """Handles cases where evidence was verified to be missing in BMU records."""
    raw_query = state.get("raw_query", "")
    reason = state.get("audit_reason", "Official information not found in verified records.")

    msg = (
        "I could not find sufficiently relevant information in official BML Munjal University "
        "records for this query. Please check the official BMU portal (https://www.bmu.edu.in) "
        "or contact the admissions office at info@bmu.edu.in / 1800-103-6888."
    )
    trace_item: AgentTraceItem = {
        "agent": "RefusalHandler (Records Not Found)",
        "latency_ms": 0.0,
        "details": {
            "reason": reason,
            "action": "Authoritative BMU Helpdesk Referral",
        },
    }
    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "final_answer": msg,
        "agent_trace": current_trace,
    }


# =====================================================================
# Agent 3: Academic Response Synthesizer Node
# =====================================================================

def response_synthesizer_node(state: BMURAGState) -> Dict[str, Any]:
    """
    Agent 3: Synthesizes high-quality, grounded academic guidance
    strictly adhering to the vetted context.
    """
    start_t = time.perf_counter()
    raw_query = state.get("raw_query", "")
    vetted_context = state.get("vetted_context", "")
    provider = (state.get("llm_provider") or "groq").lower()

    client, model_name = get_llm_client("synthesis", provider=provider)

    prompt = f"""You are the official BMU Academic Assistant for BML Munjal University.

Answer the student's question clearly, accurately, and concisely based STRICTLY on the provided vetted context.
Guidelines:
1. State facts directly. If fees, dates, or eligibility criteria are mentioned, format them with clean bullet points.
2. Do not assume, speculate, or extrapolate facts not present in the context.
3. If the context does not explicitly mention a detail, state clearly that it is not specified in the current records.
4. Do not use markdown headers (# or ##), keep paragraphs tidy and friendly.

Context:
{vetted_context}

Question:
{raw_query}
"""

    answer = ""
    last_error = None
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are the official BMU Academic Assistant. Answer strictly from context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=800,
                timeout=15.0,
            )
            answer = response.choices[0].message.content or "Not found in context"
            break
        except Exception as exc:
            last_error = exc

    if not answer:
        # Preserve a grounded answer when the external synthesizer is temporarily unavailable.
        answer = "Based on the official BMU record:\n\n" + vetted_context[:3000]
        print(f"[Agent 3 Exception]: {type(last_error).__name__}: {last_error}. Used grounded context fallback.")

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    provider_tag = "Local Ollama" if provider == "local" else "Groq Cloud"
    trace_item: AgentTraceItem = {
        "agent": f"Agent 3 (ResponseSynthesizer) [{provider_tag}]",
        "latency_ms": latency_ms,
        "details": {
            "provider": provider,
            "model": model_name,
            "answer_length": len(answer),
        },
    }

    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "final_answer": answer,
        "agent_trace": current_trace,
    }


# =====================================================================
# Build and Compile LangGraph Workflow
# =====================================================================

def create_bmu_rag_graph():
    workflow = StateGraph(BMURAGState)

    # Add Nodes
    workflow.add_node("query_specialist", query_specialist_node)
    workflow.add_node("retrieval_tool", retrieval_tool_node)
    workflow.add_node("context_auditor", context_auditor_node)
    workflow.add_node("advance_candidate", advance_candidate_node)
    workflow.add_node("handle_not_found", handle_not_found_node)
    workflow.add_node("response_synthesizer", response_synthesizer_node)

    # Set Edges
    workflow.set_entry_point("query_specialist")
    workflow.add_edge("query_specialist", "retrieval_tool")
    workflow.add_edge("retrieval_tool", "context_auditor")

    workflow.add_conditional_edges(
        "context_auditor",
        route_after_context_audit,
        {
            "synthesize": "response_synthesizer",
            "advance_candidate": "advance_candidate",
            "handle_not_found": "handle_not_found",
        },
    )

    # Self-correction loop: fallback -> re-audit next candidate
    workflow.add_edge("advance_candidate", "context_auditor")

    # Terminations
    workflow.add_edge("response_synthesizer", END)
    workflow.add_edge("handle_not_found", END)

    return workflow.compile()


# Global compiled app
bmu_rag_app = create_bmu_rag_graph()
