"""
run_100_evaluation.py
Comprehensive 100-Question Evaluation Framework for BMU Website Chatbot.
Performs side-by-side benchmark comparison between Cloud LLM (Groq) and Local LLM (Ollama).

Features:
- Rate limit mitigation (adaptive delay, jitter, 429 exponential backoff)
- Real local LLM verification (Ollama on localhost:11434)
- 20 academic & adversarial categories x 5 questions = 100 questions
- RAGAS-style evaluation: Relevance, Faithfulness, Completeness, Conciseness, Correctness
- Guardrail precision/recall validation (Prompt injection & scope safety)
- Latency statistics (Median, P95, Mean, Min, Max)
- Checkpointing after every query with resume support (--resume)
- Dry-run mode (--dry-run) and sample limits (--limit N)
- JSON and CSV output exports
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Load local environment without exposing secrets
load_dotenv()

from evaluate_metrics import evaluate_answer


API_BASE_URL = os.getenv("BMU_BACKEND_URL", "http://127.0.0.1:8000")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b")


def check_rate_limits() -> Dict[str, Any]:
    """Inspects Groq and backend rate limits safely without exposing API keys."""
    key = os.getenv("GROQ_API_KEY_QUERY_AGENT") or os.getenv("GROQ_API_KEY") or ""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    limits: Dict[str, Any] = {
        "groq_requests_limit": None,
        "groq_tokens_limit": None,
        "groq_remaining_requests": None,
        "groq_remaining_tokens": None,
        "groq_reset_requests": None,
        "groq_reset_tokens": None,
        "recommended_delay_sec": 1.5,
    }

    if not key:
        return limits

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": os.getenv("GROQ_MODEL_QUERY_AGENT", "openai/gpt-oss-20b"),
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=8,
        )
        for h, v in resp.headers.items():
            hl = h.lower()
            if "ratelimit-limit-requests" in hl:
                limits["groq_requests_limit"] = v
            elif "ratelimit-limit-tokens" in hl:
                limits["groq_tokens_limit"] = v
            elif "ratelimit-remaining-requests" in hl:
                limits["groq_remaining_requests"] = v
            elif "ratelimit-remaining-tokens" in hl:
                limits["groq_remaining_tokens"] = v
            elif "ratelimit-reset-requests" in hl:
                limits["groq_reset_requests"] = v
            elif "ratelimit-reset-tokens" in hl:
                limits["groq_reset_tokens"] = v

        # If token remaining is tight (< 2000), space calls further apart
        rem_tok = int(limits["groq_remaining_tokens"] or 8000)
        if rem_tok < 2000:
            limits["recommended_delay_sec"] = 3.0
        elif rem_tok < 4000:
            limits["recommended_delay_sec"] = 2.0
    except Exception as exc:
        print(f"[Rate Limit Check Note]: Could not poll headers: {exc}")

    return limits


def verify_services() -> Dict[str, bool]:
    """Verifies that the backend and local LLM services are operational."""
    status = {"backend": False, "local_llm": False}
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        status["backend"] = resp.status_code == 200
    except Exception:
        status["backend"] = False

    try:
        # Check Ollama tags endpoint
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            status["local_llm"] = any(LOCAL_LLM_MODEL in m for m in models)
    except Exception:
        status["local_llm"] = False

    return status


def call_chat_api(
    query: str,
    llm_provider: str = "groq",
    enable_guardrails: bool = True,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Dict[str, Any]:
    """Invokes the BMU Chatbot backend with retries and exponential backoff."""
    payload = {
        "query": query,
        "llm_provider": llm_provider,
        "enable_guardrails": enable_guardrails,
    }

    last_error = ""
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{API_BASE_URL}/ask",
                json=payload,
                timeout=120,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            if resp.status_code == 200:
                data = resp.json()
                data["request_latency_ms"] = elapsed_ms
                return data
            elif resp.status_code == 429:
                wait_sec = (2 ** attempt) * base_delay + random.uniform(0.5, 1.5)
                print(f"[429 Rate Limit]: Retrying in {wait_sec:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_sec)
                continue
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:150]}"
        except requests.RequestException as exc:
            last_error = str(exc)
            wait_sec = (2 ** attempt) * 1.5
            time.sleep(wait_sec)

    return {
        "query": query,
        "answer": f"ERROR: {last_error}",
        "top_results": [],
        "guardrail": {"status": "error", "action": "error", "reason": last_error},
        "agent_trace": [],
        "request_latency_ms": 0.0,
        "llm_provider": llm_provider,
        "guardrails_enabled": enable_guardrails,
    }


def compute_key_facts_coverage(answer: str, key_facts: List[str]) -> float:
    """Computes exact and token coverage of ground-truth key facts."""
    if not key_facts:
        return 1.0
    ans_lower = (answer or "").lower()
    covered = sum(1 for fact in key_facts if fact.lower() in ans_lower)
    return round(covered / len(key_facts), 3)


def evaluate_single_response(
    question: Dict[str, Any],
    api_response: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculates all RAGAS metrics, guardrail decisions, and latency stats for a response."""
    ans = api_response.get("answer", "")
    gt = question.get("reference_ground_truth", "")
    key_facts = question.get("key_facts", [])
    top_results = api_response.get("top_results", [])
    guardrail = api_response.get("guardrail") or {}
    agent_trace = api_response.get("agent_trace", [])

    # Approximate context from top search results
    context = ""
    if top_results:
        context = " ".join([r.get("url", "") for r in top_results])

    # Guardrail metrics
    guard_status = guardrail.get("status", "none")
    expected_action = question.get("expected_guardrail_action", "allow")
    is_blocked = guard_status == "blocked"
    is_expected_blocked = expected_action == "block"

    guardrail_correct = is_blocked == is_expected_blocked

    # For blocked queries, answer is the rejection message which is correct
    if is_blocked and is_expected_blocked:
        metrics_dict = {
            "relevance": 5.0,
            "faithfulness": 5.0,
            "completeness": 5.0,
            "conciseness": 5.0,
            "correctness": 5.0,
            "final_score": 5.0,
        }
        facts_cov = 1.0
    else:
        eval_res = evaluate_answer(
            context=context or gt,
            question=question["query"],
            answer=ans,
            ground_truth=gt,
        )
        metrics_dict = asdict(eval_res)
        facts_cov = compute_key_facts_coverage(ans, key_facts)
        # Blend fact coverage into completeness & correctness
        metrics_dict["completeness"] = round(0.5 * metrics_dict["completeness"] + 0.5 * (facts_cov * 5.0), 2)
        metrics_dict["correctness"] = round(0.6 * metrics_dict["correctness"] + 0.4 * (facts_cov * 5.0), 2)
        metrics_dict["final_score"] = round(mean(metrics_dict.values()), 2)

    retrieval_hit = len(top_results) > 0 and float(top_results[0].get("score", 0)) > 0.35
    top_score = float(top_results[0].get("score", 0)) if top_results else 0.0

    # Extract per-agent latencies
    trace_latencies = {}
    for step in agent_trace:
        agent_name = step.get("agent", "unknown")
        trace_latencies[agent_name] = step.get("latency_ms", 0.0)

    return {
        "answer": ans,
        "latency_ms": api_response.get("request_latency_ms", 0.0),
        "metrics": metrics_dict,
        "key_facts_coverage": facts_cov,
        "guardrail_status": guard_status,
        "guardrail_action": guardrail.get("action", "unknown"),
        "guardrail_reason": guardrail.get("reason", ""),
        "guardrail_accurate": guardrail_correct,
        "retrieval_hit": retrieval_hit,
        "retrieval_score": round(top_score, 4),
        "retrieved_urls": [r.get("url") for r in top_results[:3]],
        "agent_trace_latencies": trace_latencies,
        "has_error": ans.startswith("ERROR:"),
    }


def aggregate_model_metrics(rows: List[Dict[str, Any]], provider: str) -> Dict[str, Any]:
    """Aggregates all evaluation results into high-level KPIs and category breakdowns."""
    valid_rows = [r for r in rows if provider in r and not r[provider]["has_error"]]
    if not valid_rows:
        return {}

    latencies = [r[provider]["latency_ms"] for r in valid_rows]
    latencies.sort()
    med_lat = round(median(latencies), 1) if latencies else 0.0
    p95_lat = round(latencies[int(len(latencies) * 0.95)], 1) if latencies else 0.0

    scores = [r[provider]["metrics"]["final_score"] for r in valid_rows]
    relevance = [r[provider]["metrics"]["relevance"] for r in valid_rows]
    faithfulness = [r[provider]["metrics"]["faithfulness"] for r in valid_rows]
    completeness = [r[provider]["metrics"]["completeness"] for r in valid_rows]
    conciseness = [r[provider]["metrics"]["conciseness"] for r in valid_rows]
    correctness = [r[provider]["metrics"]["correctness"] for r in valid_rows]
    facts_cov = [r[provider]["key_facts_coverage"] for r in valid_rows]

    guard_acc = [1.0 if r[provider]["guardrail_accurate"] else 0.0 for r in valid_rows]
    retrieval_hits = [1.0 if r[provider]["retrieval_hit"] else 0.0 for r in valid_rows]

    # Category breakdowns
    categories: Dict[str, List[float]] = {}
    for r in valid_rows:
        cat = r["category"]
        categories.setdefault(cat, []).append(r[provider]["metrics"]["final_score"])

    cat_breakdown = {cat: round(mean(sc), 2) for cat, sc in categories.items()}

    return {
        "total_evaluated": len(valid_rows),
        "total_errors": len([r for r in rows if provider in r and r[provider]["has_error"]]),
        "avg_final_score": round(mean(scores), 2) if scores else 0.0,
        "avg_final_percentage": round((mean(scores) / 5.0) * 100, 1) if scores else 0.0,
        "avg_relevance": round(mean(relevance), 2) if relevance else 0.0,
        "avg_faithfulness": round(mean(faithfulness), 2) if faithfulness else 0.0,
        "avg_completeness": round(mean(completeness), 2) if completeness else 0.0,
        "avg_conciseness": round(mean(conciseness), 2) if conciseness else 0.0,
        "avg_correctness": round(mean(correctness), 2) if correctness else 0.0,
        "avg_key_facts_coverage": round(mean(facts_cov), 3) if facts_cov else 0.0,
        "guardrail_accuracy": round(mean(guard_acc), 3) if guard_acc else 0.0,
        "retrieval_hit_rate": round(mean(retrieval_hits), 3) if retrieval_hits else 0.0,
        "latency_median_ms": med_lat,
        "latency_p95_ms": p95_lat,
        "latency_mean_ms": round(mean(latencies), 1) if latencies else 0.0,
        "latency_min_ms": round(min(latencies), 1) if latencies else 0.0,
        "latency_max_ms": round(max(latencies), 1) if latencies else 0.0,
        "category_scores": cat_breakdown,
    }


def save_csv_export(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    """Exports question-by-question comparative metrics to CSV format."""
    fieldnames = [
        "id",
        "category",
        "difficulty",
        "expected_guardrail",
        "query",
        "groq_score",
        "groq_relevance",
        "groq_faithfulness",
        "groq_correctness",
        "groq_latency_ms",
        "groq_guardrail",
        "local_score",
        "local_relevance",
        "local_faithfulness",
        "local_correctness",
        "local_latency_ms",
        "local_guardrail",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            g = r.get("groq", {})
            l = r.get("local", {})
            gm = g.get("metrics", {})
            lm = l.get("metrics", {})
            writer.writerow({
                "id": r.get("id"),
                "category": r.get("category"),
                "difficulty": r.get("difficulty"),
                "expected_guardrail": r.get("expected_guardrail_action"),
                "query": r.get("query"),
                "groq_score": gm.get("final_score", ""),
                "groq_relevance": gm.get("relevance", ""),
                "groq_faithfulness": gm.get("faithfulness", ""),
                "groq_correctness": gm.get("correctness", ""),
                "groq_latency_ms": g.get("latency_ms", ""),
                "groq_guardrail": g.get("guardrail_status", ""),
                "local_score": lm.get("final_score", ""),
                "local_relevance": lm.get("relevance", ""),
                "local_faithfulness": lm.get("faithfulness", ""),
                "local_correctness": lm.get("correctness", ""),
                "local_latency_ms": l.get("latency_ms", ""),
                "local_guardrail": l.get("guardrail_status", ""),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="BMU Chatbot 100-Question Dual-Model Evaluation Framework")
    parser.add_argument("--provider", choices=["groq", "local", "both"], default="both", help="Which LLM provider to evaluate")
    parser.add_argument("--limit", type=int, default=100, help="Number of questions to evaluate (1..100)")
    parser.add_argument("--delay", type=float, default=1.8, help="Inter-query delay in seconds to preserve Groq TPM")
    parser.add_argument("--dry-run", action="store_true", help="Execute 3 sample queries to verify pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint file")
    args = parser.parse_args()

    output_dir = Path("evaluation_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "evaluation_100_checkpoint.json"
    benchmark_file = output_dir / "benchmark_100_questions.json"

    # 1. Load benchmark questions
    if not benchmark_file.exists():
        from benchmark_100_questions import BENCHMARK_DATA, generate_benchmark_file
        generate_benchmark_file(benchmark_file)

    with open(benchmark_file, "r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    if args.dry_run:
        print("[Dry Run Mode Activated]: Evaluating 3 representative sample questions.")
        # Pick 1 legitimate, 1 out-of-scope, 1 adversarial
        questions = [questions[0], questions[90], questions[95]]
    else:
        questions = questions[:args.limit]

    # 2. Service & Rate Limit Preflight Checks
    print("\n" + "=" * 60)
    print("BMU Chatbot 100-Question Benchmark Framework")
    print("=" * 60)
    svc_status = verify_services()
    print(f"Backend Status (127.0.0.1:8000): {'[ONLINE]' if svc_status['backend'] else '[OFFLINE]'}")
    print(f"Local LLM Status ({LOCAL_LLM_MODEL}): {'[ONLINE]' if svc_status['local_llm'] else '[OFFLINE]'}")

    if not svc_status["backend"]:
        print("ERROR: BMU Backend on 127.0.0.1:8000 is not responding! Start it before evaluating.")
        sys.exit(1)

    rate_info = check_rate_limits()
    inter_delay = max(args.delay, rate_info.get("recommended_delay_sec", 1.8))
    print(f"Groq Rate Limits: Remaining Requests={rate_info.get('groq_remaining_requests')}, Remaining Tokens={rate_info.get('groq_remaining_tokens')}")
    print(f"Safe Pacing Delay: {inter_delay:.1f}s between queries")

    # 3. Checkpoint & Resume
    results_map: Dict[int, Dict[str, Any]] = {}
    if (args.resume or checkpoint_path.exists()) and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results_map = {item["id"]: item for item in saved}
            print(f"[Resume]: Loaded {len(results_map)} completed evaluations from checkpoint.", flush=True)
        except Exception as exc:
            print(f"[Resume Warning]: Failed to load checkpoint: {exc}", flush=True)

    # 4. Evaluation Execution
    providers_to_test = ["groq", "local"] if args.provider == "both" else [args.provider]
    if "local" in providers_to_test and not svc_status["local_llm"]:
        print("[Warning]: Local LLM (Ollama) is not reachable! Local tests will be marked with fallback notice.", flush=True)

    total_q = len(questions)
    for idx, q in enumerate(questions, start=1):
        q_id = q["id"]
        row: Dict[str, Any] = results_map.get(q_id, {
            "id": q_id,
            "category": q["category"],
            "difficulty": q["difficulty"],
            "expected_guardrail_action": q["expected_guardrail_action"],
            "expected_scope": q["expected_scope"],
            "query": q["query"],
            "reference_ground_truth": q["reference_ground_truth"],
        })

        print(f"\n[{idx}/{total_q}] Q{q_id} [{q['category']}] ({q['difficulty']})", flush=True)
        print(f" Query: {q['query']}", flush=True)

        # Groq Run
        if "groq" in providers_to_test:
            if "groq" in row and not row["groq"].get("has_error"):
                print(f"  -> Groq: Skipped (Loaded from checkpoint, Score: {row['groq']['metrics']['final_score']}/5.0)", flush=True)
            else:
                resp_groq = call_chat_api(q["query"], llm_provider="groq", enable_guardrails=True)
                eval_groq = evaluate_single_response(q, resp_groq)
                row["groq"] = eval_groq
                print(f"  -> Groq: Score={eval_groq['metrics']['final_score']}/5.0 | Latency={eval_groq['latency_ms']}ms | Guardrail={eval_groq['guardrail_status']}", flush=True)
                time.sleep(inter_delay)

        # Local LLM Run
        if "local" in providers_to_test:
            if "local" in row and not row["local"].get("has_error"):
                print(f"  -> Local LLM: Skipped (Loaded from checkpoint, Score: {row['local']['metrics']['final_score']}/5.0)", flush=True)
            else:
                resp_local = call_chat_api(q["query"], llm_provider="local", enable_guardrails=True)
                eval_local = evaluate_single_response(q, resp_local)
                row["local"] = eval_local
                print(f"  -> Local: Score={eval_local['metrics']['final_score']}/5.0 | Latency={eval_local['latency_ms']}ms | Guardrail={eval_local['guardrail_status']}", flush=True)
                # Local LLM doesn't hit Groq rate limits, so brief delay is sufficient
                time.sleep(0.5)

        results_map[q_id] = row

        # Checkpoint save after every question
        checkpoint_rows = sorted(results_map.values(), key=lambda x: x["id"])
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_rows, f, indent=2, ensure_ascii=False)

    # 5. Final Aggregation & Summarization
    final_rows = sorted(results_map.values(), key=lambda x: x["id"])
    groq_summary = aggregate_model_metrics(final_rows, "groq") if "groq" in providers_to_test else {}
    local_summary = aggregate_model_metrics(final_rows, "local") if "local" in providers_to_test else {}

    summary = {
        "metadata": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": len(final_rows),
            "backend_url": API_BASE_URL,
            "groq_model": os.getenv("GROQ_MODEL_SYNTHESIS_AGENT", "openai/gpt-oss-120b"),
            "local_model": LOCAL_LLM_MODEL,
        },
        "groq_summary": groq_summary,
        "local_summary": local_summary,
    }

    # Save final JSON and CSV
    final_json_path = output_dir / "evaluation_100_results.json"
    summary_json_path = output_dir / "evaluation_100_summary.json"
    csv_path = output_dir / "evaluation_100_results.csv"

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, indent=2, ensure_ascii=False)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_csv_export(final_rows, csv_path)

    # Print clean comparison table to console
    print("\n" + "=" * 65)
    print("             BMU EVALUATION BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<28} | {'Cloud LLM (Groq)':<16} | {'Local LLM (Ollama)':<16}")
    print("-" * 65)
    print(f"{'Overall Score (0-5)':<28} | {groq_summary.get('avg_final_score', 'N/A'):<16} | {local_summary.get('avg_final_score', 'N/A'):<16}")
    print(f"{'Quality Percentage':<28} | {str(groq_summary.get('avg_final_percentage', 'N/A')) + '%':<16} | {str(local_summary.get('avg_final_percentage', 'N/A')) + '%':<16}")
    print(f"{'Relevance':<28} | {groq_summary.get('avg_relevance', 'N/A'):<16} | {local_summary.get('avg_relevance', 'N/A'):<16}")
    print(f"{'Faithfulness':<28} | {groq_summary.get('avg_faithfulness', 'N/A'):<16} | {local_summary.get('avg_faithfulness', 'N/A'):<16}")
    print(f"{'Correctness':<28} | {groq_summary.get('avg_correctness', 'N/A'):<16} | {local_summary.get('avg_correctness', 'N/A'):<16}")
    print(f"{'Guardrail Accuracy':<28} | {str(round(groq_summary.get('guardrail_accuracy', 0)*100, 1)) + '%':<16} | {str(round(local_summary.get('guardrail_accuracy', 0)*100, 1)) + '%':<16}")
    print(f"{'Median Latency':<28} | {str(groq_summary.get('latency_median_ms', 'N/A')) + 'ms':<16} | {str(local_summary.get('latency_median_ms', 'N/A')) + 'ms':<16}")
    print(f"{'P95 Latency':<28} | {str(groq_summary.get('latency_p95_ms', 'N/A')) + 'ms':<16} | {str(local_summary.get('latency_p95_ms', 'N/A')) + 'ms':<16}")
    print(f"{'Total Evaluated':<28} | {groq_summary.get('total_evaluated', 'N/A'):<16} | {local_summary.get('total_evaluated', 'N/A'):<16}")
    print("=" * 65)
    print(f"Results saved to:\n- {final_json_path}\n- {summary_json_path}\n- {csv_path}\n")


if __name__ == "__main__":
    main()
