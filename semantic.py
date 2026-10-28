import json
import os
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jev_guardrails import jev_guardrail
from bmu_rag_graph import bmu_rag_app, BMURAGState
from phoenix_tracer import init_phoenix_tracing, get_tracer

load_dotenv()
init_phoenix_tracing()

app = FastAPI(title="BMU Multi-Agent Semantic QA API (LangGraph + JEV)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    llm_provider: Optional[str] = Field("groq", description="groq | local")
    enable_guardrails: Optional[bool] = Field(True, description="Whether JEV guardrails are active")


class RankedUrl(BaseModel):
    url: str
    score: float


class GuardrailMeta(BaseModel):
    status: str = Field(..., description="passed | blocked | flagged | bypassed")
    action: str = Field(..., description="allow | block | flag_disclaimer | retrieval_gated | disabled")
    reason: str
    engine: str = "jev"
    input_checks: dict[str, Any] = Field(default_factory=dict)
    output_checks: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    query: str
    answer: str
    top_results: list[RankedUrl]
    guardrail: Optional[GuardrailMeta] = None
    agent_trace: Optional[list[dict[str, Any]]] = None
    llm_provider: str = "groq"
    guardrails_enabled: bool = True


def _generate_unconstrained_answer(query: str, provider: str = "groq") -> str:
    from bmu_rag_graph import get_llm_client
    client, model_name = get_llm_client("synthesis", provider=provider)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful, friendly AI assistant. Answer the user's request thoroughly, accurately, and directly.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content or "No response generated."
    except Exception as exc:
        return f"Unable to generate unconstrained response: {exc}"


def _build_answer_internal(
    query: str,
    llm_provider: str = "groq",
    enable_guardrails: bool = True,
) -> AskResponse:
    provider = (llm_provider or "groq").lower()
    if provider not in ["groq", "local"]:
        provider = "groq"

    # =========================================================================
    # 1. JEV INPUT GUARDRAILS (Prompt Injection, Toxicity, Academic Domain)
    # =========================================================================
    input_category = "general"
    input_guard = None

    if enable_guardrails:
        input_guard = jev_guardrail.validate_input(query)
        if not input_guard.allowed:
            return AskResponse(
                query=query,
                answer=input_guard.suggested_response or "Query blocked by security guardrails.",
                top_results=[],
                guardrail=GuardrailMeta(
                    status="blocked",
                    action="block",
                    reason=input_guard.reason,
                    engine=input_guard.engine,
                    input_checks={
                        "category": input_guard.category,
                        "injection_probability": input_guard.injection_probability,
                        "toxicity_probability": input_guard.toxicity_probability,
                        "latency_ms": input_guard.latency_ms,
                    },
                ),
                agent_trace=[],
                llm_provider=provider,
                guardrails_enabled=True,
            )
        input_category = input_guard.category
    else:
        # Heuristic intent inference when guardrails are toggled off
        q_lower = query.lower()
        if any(k in q_lower for k in ["fee", "fees", "cost", "scholarship"]):
            input_category = "fees"
        elif any(k in q_lower for k in ["eligibility", "admission", "criteria", "apply"]):
            input_category = "admissions"
        elif any(k in q_lower for k in ["b.tech", "cse", "mba", "course", "program"]):
            input_category = "courses"

    # =========================================================================
    # 2. LANGGRAPH MULTI-AGENT WORKFLOW
    #    Agent 1 (QuerySpecialist) -> HybridRetrievalTool -> Agent 2 (ContextAuditor)
    #    [Autonomous Loop] -> Agent 3 (ResponseSynthesizer)
    # =========================================================================
    initial_state: BMURAGState = {
        "raw_query": query,
        "llm_provider": provider,
        "optimized_query": query,
        "keywords": [],
        "intent_category": input_category,
        "candidates": [],
        "current_candidate_idx": 0,
        "audit_decision": "TRY_FALLBACK",
        "audit_reason": "",
        "vetted_context": "",
        "final_answer": "",
        "top_results": [],
        "agent_trace": [],
        "error": None,
    }

    graph_result = bmu_rag_app.invoke(initial_state)
    raw_answer = graph_result.get("final_answer", "Not found in context")
    top_results_data = graph_result.get("top_results", [])
    vetted_context = graph_result.get("vetted_context", "")
    agent_trace = graph_result.get("agent_trace", [])

    ranked_urls = [
        RankedUrl(url=item["url"], score=float(item["score"]))
        for item in top_results_data
        if item.get("url")
    ]

    # If guardrails are toggled OFF: answer freely even if outside BMU institutional context
    if not enable_guardrails:
        if graph_result.get("audit_decision") == "NOT_FOUND" or not vetted_context:
            raw_answer = _generate_unconstrained_answer(query, provider)
            agent_trace.append({
                "agent": f"UnconstrainedSynthesizer [{provider.upper()}]",
                "latency_ms": 0.0,
                "details": {
                    "note": "Guardrails OFF: answered general/out-of-scope inquiry directly without refusal",
                    "provider": provider,
                },
            })

        return AskResponse(
            query=query,
            answer=raw_answer,
            top_results=ranked_urls,
            guardrail=GuardrailMeta(
                status="bypassed",
                action="disabled",
                reason="JEV Guardrails disabled by user toggle (raw model output)",
                engine="none",
                input_checks={"guardrails_enabled": False},
                output_checks={"guardrails_enabled": False},
            ),
            agent_trace=agent_trace,
            llm_provider=provider,
            guardrails_enabled=False,
        )

    # If evidence was not found, return authoritative message without output guard check
    if graph_result.get("audit_decision") == "NOT_FOUND" or not vetted_context:
        return AskResponse(
            query=query,
            answer=raw_answer,
            top_results=ranked_urls,
            guardrail=GuardrailMeta(
                status="passed",
                action="retrieval_gated",
                reason=graph_result.get("audit_reason", "Official information not found in verified records."),
                engine=input_guard.engine if input_guard else "jev",
                input_checks={
                    "category": input_guard.category if input_guard else input_category,
                    "latency_ms": input_guard.latency_ms if input_guard else 0.0,
                },
            ),
            agent_trace=agent_trace,
            llm_provider=provider,
            guardrails_enabled=True,
        )

    # =========================================================================
    # 3. JEV OUTPUT GUARDRAILS (Grounding / Hallucination & Safety Verification)
    # =========================================================================
    source_url = ranked_urls[0].url if ranked_urls else ""
    output_guard = jev_guardrail.validate_output(
        query=query,
        context=vetted_context,
        answer=raw_answer,
        source_url=source_url,
    )

    return AskResponse(
        query=query,
        answer=output_guard.final_answer,
        top_results=ranked_urls,
        guardrail=GuardrailMeta(
            status="flagged" if output_guard.action == "flag_disclaimer" else ("blocked" if output_guard.action == "block" else "passed"),
            action=output_guard.action,
            reason="Verified grounded in context" if output_guard.is_grounded else "Low context grounding detected; verification disclaimer appended",
            engine=output_guard.engine,
            input_checks={
                "category": input_guard.category if input_guard else input_category,
                "latency_ms": input_guard.latency_ms if input_guard else 0.0,
            },
            output_checks={
                "is_grounded": output_guard.is_grounded,
                "grounding_probability": output_guard.grounding_probability,
                "is_safe": output_guard.is_safe,
                "latency_ms": output_guard.latency_ms,
            },
        ),
        agent_trace=agent_trace,
        llm_provider=provider,
        guardrails_enabled=True,
    )


def _build_answer(
    query: str,
    llm_provider: str = "groq",
    enable_guardrails: bool = True,
) -> AskResponse:
    tracer = get_tracer()
    provider = (llm_provider or "groq").lower()
    try:
        with tracer.start_as_current_span("bmu_rag_pipeline") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.value", query)

            # High-visibility Phoenix attributes for filtering & observability
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.is_local", provider == "local")
            span.set_attribute("guardrails.enabled", enable_guardrails)
            span.set_attribute(
                "response.source",
                f"LLM:{provider.upper()} | Guardrails:{'ON' if enable_guardrails else 'OFF'}",
            )
            span.set_attribute(
                "execution.config",
                f"Provider={provider.upper()}, Guardrails={'Enabled' if enable_guardrails else 'Disabled'}",
            )

            resp = _build_answer_internal(
                query=query,
                llm_provider=provider,
                enable_guardrails=enable_guardrails,
            )
            span.set_attribute("output.value", resp.answer)
            span.set_attribute("llm.provider_used", resp.llm_provider)
            span.set_attribute("guardrails.status_used", resp.guardrails_enabled)

            if resp.guardrail:
                span.set_attribute("guardrail.status", resp.guardrail.status)
                span.set_attribute("guardrail.action", resp.guardrail.action)
                span.set_attribute("guardrail.reason", resp.guardrail.reason)
                span.set_attribute("guardrail.engine", resp.guardrail.engine)
            return resp
    except Exception:
        # Fallback to direct execution if tracing encounters any issue
        return _build_answer_internal(
            query=query,
            llm_provider=provider,
            enable_guardrails=enable_guardrails,
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents/status")
def agents_status() -> dict[str, Any]:
    return {
        "status": "active",
        "orchestrator": "LangGraph StateGraph",
        "supported_llm_providers": ["groq", "local"],
        "local_llm": {
            "url": os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
            "model": os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b"),
        },
        "agents": [
            {
                "name": "Agent 1: Query Specialist",
                "role": "Academic terminology expansion & query vector optimization",
                "model": os.getenv("GROQ_MODEL_QUERY_AGENT", "openai/gpt-oss-20b"),
                "key_configured": bool(os.getenv("GROQ_API_KEY_QUERY_AGENT")),
            },
            {
                "name": "Deterministic Tool: Hybrid Retrieval & Extractor",
                "role": "SentenceTransformer dense embeddings & ChromaDB hybrid search",
                "type": "Standalone FastAPI Retrieval Microservice",
                "endpoint": os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8001"),
            },
            {
                "name": "Agent 2: Context Auditor",
                "role": "Information sufficiency auditor & autonomous fallback decider",
                "model": os.getenv("GROQ_MODEL_CONTEXT_AGENT", "openai/gpt-oss-20b"),
                "key_configured": bool(os.getenv("GROQ_API_KEY_CONTEXT_AGENT")),
            },
            {
                "name": "Agent 3: Academic Response Synthesizer",
                "role": "High-capacity grounded academic guidance generator",
                "model": os.getenv("GROQ_MODEL_SYNTHESIS_AGENT", "openai/gpt-oss-120b"),
                "key_configured": bool(os.getenv("GROQ_API_KEY_SYNTHESIS_AGENT")),
            },
        ],
    }


@app.get("/guardrails/status")
def guardrails_status() -> dict[str, Any]:
    return {
        "status": "active",
        "has_live_jev_client": jev_guardrail.has_live_client,
        "engine": "jev-cloud" if jev_guardrail.has_live_client else "jev-system-one-local",
        "active_guardrails": [
            "Input Prompt Injection Detection (Noul)",
            "Input Content Toxicity & Abuse Detection (Noul)",
            "Input Academic Domain Scope Classification (Choice)",
            "Retrieval Score Cutoff Gating (Deterministic)",
            "Output Factual Grounding & Hallucination Guardrail (Noul)",
            "Output Content Safety Verification (Noul)",
        ],
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return _build_answer(
            query=request.query,
            llm_provider=request.llm_provider or "groq",
            enable_guardrails=True if request.enable_guardrails is None else request.enable_guardrails,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/evaluation/results")
def get_evaluation_results() -> Any:
    results_path = os.path.join("evaluation_outputs", "evaluation_100_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/evaluation/summary")
def get_evaluation_summary() -> Any:
    summary_path = os.path.join("evaluation_outputs", "evaluation_100_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.get("/evaluation/benchmark")
def get_evaluation_benchmark() -> Any:
    bench_path = os.path.join("evaluation_outputs", "benchmark_100_questions.json")
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/evaluation/export/csv")
def export_evaluation_csv() -> Any:
    csv_path = os.path.join("evaluation_outputs", "evaluation_100_results.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="bmu_evaluation_100_results.csv")
    raise HTTPException(status_code=404, detail="CSV report not generated yet.")


# Mount frontend directory for browser dashboard
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/dashboard", StaticFiles(directory=frontend_dir, html=True), name="dashboard")


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/dashboard/")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting BMU Semantic Orchestrator on http://{host}:{port}...")
    uvicorn.run("semantic:app", host=host, port=port, reload=False)