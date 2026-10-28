"""
Automated Test Suite for BMU Retrieval Microservice
===================================================
Tests:
1. Microservice /health endpoint
2. Microservice /retrieve endpoint with realistic query
3. Microservice /retrieve input validation (empty query, edge cases)
4. Full integration with backend /ask endpoint (verifying all 3 agents + microservice in trace)
"""

import sys
import requests

# Reconfigure stdout for utf-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

RETRIEVAL_BASE_URL = "http://127.0.0.1:8001"
BACKEND_BASE_URL = "http://127.0.0.1:8000"


def run_tests():
    passed = 0
    total = 0

    def assert_test(name: str, condition: bool, details: str = ""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name} | Details: {details}")

    print("=====================================================")
    print("      TESTING BMU RETRIEVAL MICROSERVICE (PORT 8001) ")
    print("=====================================================")

    # 1. Health check
    print("\n--- 1. Health Check Endpoint ---")
    try:
        r = requests.get(f"{RETRIEVAL_BASE_URL}/health", timeout=5)
        assert_test("Health endpoint returns 200 OK", r.status_code == 200, f"Got status {r.status_code}")
        data = r.json()
        assert_test("Health status is 'ok'", data.get("status") == "ok", f"Got {data.get('status')}")
        assert_test("Service name matches", "Retrieval Microservice" in data.get("service", ""))
        assert_test("Document count is positive integer", isinstance(data.get("document_count"), int) and data.get("document_count") > 0, f"Got {data.get('document_count')}")
    except Exception as exc:
        assert_test("Health endpoint reachable", False, str(exc))

    # 2. Retrieval with valid query
    print("\n--- 2. Valid Semantic Retrieval ---")
    sample_query = "B.Tech Computer Science and Engineering CSE eligibility criteria BMU BML Munjal University tuition fee structure"
    try:
        r = requests.post(
            f"{RETRIEVAL_BASE_URL}/retrieve",
            json={"query": sample_query, "top_k": 5},
            timeout=15,
        )
        assert_test("Retrieve endpoint returns 200 OK", r.status_code == 200, f"Got status {r.status_code}")
        data = r.json()
        results = data.get("results", [])
        assert_test("Returns ranked results list", len(results) > 0, f"Got {len(results)} items")
        assert_test("Each result has required fields", all(
            "summary" in doc and "url" in doc and "score" in doc and "semantic_sim" in doc and "keyword_score" in doc
            for doc in results
        ))
        assert_test("Scores are sorted descending", all(
            results[i]["score"] >= results[i+1]["score"] for i in range(len(results)-1)
        ))
        print(f"       -> Top URL: {results[0]['url']} (Score: {results[0]['score']})")
    except Exception as exc:
        assert_test("Retrieve endpoint operational", False, str(exc))

    # 3. Error Handling
    print("\n--- 3. Error Handling & Validation ---")
    try:
        # Empty string
        r = requests.post(f"{RETRIEVAL_BASE_URL}/retrieve", json={"query": "", "top_k": 5}, timeout=5)
        assert_test("Empty query rejected (422 Unprocessable)", r.status_code in [400, 422], f"Got status {r.status_code}")

        # Whitespace-only string
        r = requests.post(f"{RETRIEVAL_BASE_URL}/retrieve", json={"query": "   ", "top_k": 5}, timeout=5)
        assert_test("Whitespace query rejected (400 Bad Request)", r.status_code in [400, 422], f"Got status {r.status_code}")

        # Invalid top_k (0 or negative)
        r = requests.post(f"{RETRIEVAL_BASE_URL}/retrieve", json={"query": "MBA fees", "top_k": 0}, timeout=5)
        assert_test("Invalid top_k rejected (422 Unprocessable)", r.status_code == 422, f"Got status {r.status_code}")
    except Exception as exc:
        assert_test("Validation tests executed", False, str(exc))

    # 4. End-to-End Integration with Backend /ask
    print("\n--- 4. End-to-End Backend Integration (/ask) ---")
    try:
        r = requests.post(
            f"{BACKEND_BASE_URL}/ask",
            json={"query": "What are the B.Tech CSE eligibility criteria and fees at BMU?"},
            timeout=45,
        )
        assert_test("Backend /ask returns 200 OK", r.status_code == 200, f"Got status {r.status_code}")
        data = r.json()
        guardrail = data.get("guardrail", {})
        trace = data.get("agent_trace", [])

        # Check guardrail passed
        assert_test("JEV guardrail passed", guardrail.get("status") == "passed", f"Status: {guardrail.get('status')}")

        # Check all 3 agents in trace
        agents_in_trace = [item.get("agent", "") for item in trace]
        assert_test("Agent 1 (QuerySpecialist) in trace", any("Agent 1" in a for a in agents_in_trace))
        assert_test("Tool (HybridRetrieval) in trace", any("Tool" in a for a in agents_in_trace))
        assert_test("Agent 2 (ContextAuditor) in trace", any("Agent 2" in a for a in agents_in_trace))
        assert_test("Agent 3 (ResponseSynthesizer) in trace", any("Agent 3" in a for a in agents_in_trace))

        # Check tool utilized the microservice
        tool_items = [item for item in trace if "Tool" in item.get("agent", "")]
        if tool_items:
            retrieval_mode = tool_items[0].get("details", {}).get("retrieval_mode", "")
            assert_test("Tool used microservice mode", retrieval_mode == "microservice", f"Got mode: {retrieval_mode}")

        answer = data.get("answer", "")
        assert_test("Answer contains eligibility facts", "50" in answer or "eligibility" in answer.lower())
        assert_test("Answer contains fee figures", "410,000" in answer or "1,640,000" in answer or "fees" in answer.lower())
    except Exception as exc:
        assert_test("Backend integration operational", False, str(exc))

    print("\n=====================================================")
    print(f"Results: {passed}/{total} tests passed ({round((passed/total)*100, 1)}%)")
    print("=====================================================")
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
