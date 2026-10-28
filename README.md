# 🎓 BMU Website Chatbot: Multi-Agent RAG with Microservices, JEV Guardrails & Observability

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![Arize Phoenix](https://img.shields.io/badge/Arize%20Phoenix-Observability-FF007A.svg)](https://phoenix.arize.com/)
[![Groq Cloud](https://img.shields.io/badge/Groq-LPU%20Inference-F55036.svg)](https://groq.com/)
[![Ollama Local](https://img.shields.io/badge/Ollama-Local%20Llama%203.2-000000.svg)](https://ollama.com/)
[![Google Stitch MCP](https://img.shields.io/badge/Stitch%20MCP-Obsidian%20Telemetry-4285F4.svg)](https://stitch.googleapis.com)

An enterprise-grade, institutional AI Chatbot tailored specifically for **BML Munjal University (BMU)**. Engineered with a decoupled **4-microservices architecture**, an autonomous 3-agent **LangGraph** state machine, a two-stage **JEV Guardrails** security and hallucination mitigation engine, dual-engine LLM runtime (**Groq Cloud LPU** vs. **Local Ollama**), end-to-end **Arize Phoenix** OpenTelemetry tracing, and a **100-Question Benchmark Observatory Dashboard** designed with **Google Stitch MCP**.

---

## 📑 Table of Contents

1. [System Architecture & Data Flow](#1-system-architecture--data-flow)
2. [Why Microservices? Deep-Dive into Each Service](#2-why-microservices-deep-dive-into-each-service)
   - [2.1 Backend Orchestrator (`semantic.py` - Port 8000)](#21-backend-orchestrator-semanticpy---port-8000)
   - [2.2 Retrieval Microservice (`retrieval_service.py` - Port 8001)](#22-retrieval-microservice-retrieval_servicepy---port-8001)
   - [2.3 Ingestion Microservice (`ingestion_service.py` - Port 8002)](#23-ingestion-microservice-ingestion_servicepy---port-8002)
   - [2.4 Arize Phoenix Observability Server (`phoenix_service.py` - Port 6006)](#24-arize-phoenix-observability-server-phoenix_servicepy---port-6006)
3. [The 3-Agent LangGraph Multi-Agent Engine](#3-the-3-agent-langgraph-multi-agent-engine)
   - [3.1 What is an Agent & Why Multi-Agent RAG?](#31-what-is-an-agent--why-multi-agent-rag)
   - [3.2 The Shared State Machine (`AgentState`)](#32-the-shared-state-machine-agentstate)
   - [3.3 Agent 1: Query Specialist](#33-agent-1-query-specialist)
   - [3.4 Deterministic Hybrid Retrieval Tool & 3.0s Fallback Safeguard](#34-deterministic-hybrid-retrieval-tool--30s-fallback-safeguard)
   - [3.5 Agent 2: Context Auditor & Self-Correction Loop](#35-agent-2-context-auditor--self-correction-loop)
   - [3.6 Agent 3: Response Synthesizer & Citation Grounding](#36-agent-3-response-synthesizer--citation-grounding)
4. [JEV Guardrails: Security, Scope & Hallucination Defense](#4-jev-guardrails-security-scope--hallucination-defense)
   - [4.1 What is JEV (Judge / Evaluate / Validate)?](#41-what-is-jev-judge--evaluate--validate)
   - [4.2 Input Guardrail Pipeline](#42-input-guardrail-pipeline)
   - [4.3 Output Guardrail Pipeline](#43-output-guardrail-pipeline)
   - [4.4 Guardrails ON vs. Guardrails OFF Bypass Mode](#44-guardrails-on-vs-guardrails-off-bypass-mode)
5. [Arize Phoenix Tracing & Observability](#5-arize-phoenix-tracing--observability)
   - [5.1 Tracing vs. Traditional Logging](#51-tracing-vs-traditional-logging)
   - [5.2 OpenTelemetry Span Tree & Metadata](#52-opentelemetry-span-tree--metadata)
   - [5.3 How to Inspect Traces in the Phoenix UI](#53-how-to-inspect-traces-in-the-phoenix-ui)
6. [Dual LLM Runtime: Cloud (Groq) vs. Local (Ollama)](#6-dual-llm-runtime-cloud-groq-vs-local-ollama)
   - [6.1 Runtime Comparison Table](#61-runtime-comparison-table)
   - [6.2 Seamless Frontend Toggle Architecture](#62-seamless-frontend-toggle-architecture)
7. [Vector Database & Embedding Pipelines](#7-vector-database--embedding-pipelines)
   - [7.1 ChromaDB Vector Store](#71-chromadb-vector-store)
   - [7.2 Dense MiniLM vs. Voyage AI Embeddings](#72-dense-minilm-vs-voyage-ai-embeddings)
   - [7.3 Hybrid Scoring Formula](#73-hybrid-scoring-formula)
8. [Web Ingestion & Extractive Summarization](#8-web-ingestion--extractive-summarization)
   - [8.1 Trafilatura Web Scraper](#81-trafilatura-web-scraper)
   - [8.2 Hybrid Extractive Summarization Algorithm](#82-hybrid-extractive-summarization-algorithm)
9. [100-Question Evaluation Framework & RAGAS Metrics](#9-100-question-evaluation-framework--ragas-metrics)
   - [9.1 Purpose of Benchmarking & Rate-Limit Strategy](#91-purpose-of-benchmarking--rate-limit-strategy)
   - [9.2 20 Categories × 5 Questions Dataset](#92-20-categories--5-questions-dataset)
   - [9.3 RAGAS Mathematical Scoring Formulas](#93-ragas-mathematical-scoring-formulas)
10. [Google Stitch MCP & The Obsidian Telemetry Observatory](#10-google-stitch-mcp--the-obsidian-telemetry-observatory)
11. [Complete Project File Map](#11-complete-project-file-map)
12. [Installation, Setup & Quickstart](#12-installation-setup--quickstart)
13. [Port & API Reference Table](#13-port--api-reference-table)
14. [Troubleshooting & FAQ](#14-troubleshooting--faq)

---

## 1. System Architecture & Data Flow

```
                                  +---------------------------------------+
                                  |         User / Student Browser        |
                                  |  - Chat UI (index.html)               |
                                  |  - Observatory UI (evaluation.html)   |
                                  +-------------------+-------------------+
                                                      |
                                          HTTP POST /ask { query,
                                            llm_provider: "groq"|"local",
                                            guardrails_enabled: true|false }
                                                      |
                                                      v
+-------------------------------------------------------------------------------------------------------+
|  PORT 8000: BACKEND ORCHESTRATOR (semantic.py)                                                        |
|                                                                                                       |
|  1. JEV INPUT GUARDRAILS (jev_guardrails.py)                                                          |
|     ├── [If Guardrails = ON]                                                                          |
|     │    ├── Check 1: Prompt Injection / Jailbreak (DAN, System Prompt Leak, SQL/Code Injection)       |
|     │    ├── Check 2: Toxicity, Harassment & Offensive Content                                        |
|     │    └── Check 3: Domain Scope Filter (Must pertain to BML Munjal University)                     |
|     └── [If Guardrails = OFF] -> Bypasses all input checks directly to LLM                            |
|                                                                                                       |
|  2. LANGGRAPH 3-AGENT STATE MACHINE (bmu_rag_graph.py)                                                |
|     ├── Agent 1: Query Specialist (Intent classification, keyword expansion)                          |
|     │       │                                                                                         |
|     │       v                                                                                         |
|     │   Deterministic Retrieval Tool                                                                  |
|     │       │──> HTTP POST http://127.0.0.1:8001/retrieve  (Retrieval Microservice)                    |
|     │       └──> 3.0s Live Scrape Timeout -> Instant Fallback to Cached ChromaDB Knowledge           |
|     │       │                                                                                         |
|     │       v                                                                                         |
|     │   Agent 2: Context Auditor                                                                      |
|     │       ├── Evaluates document sufficiency: [SUFFICIENT | TRY_FALLBACK | NOT_FOUND]               |
|     │       └── Loops to Candidate #2 / Synthesized Cluster if primary is incomplete                  |
|     │       │                                                                                         |
|     │       v                                                                                         |
|     │   Agent 3: Response Synthesizer                                                                 |
|     │       ├── Formats clear Markdown response grounded strictly in audited context                  |
|     │       └── Generates official institutional citations with verified source URLs                  |
|     │                                                                                                 |
|  3. JEV OUTPUT GUARDRAILS                                                                             |
|     ├── Check 1: Hallucination & Fact Grounding (Containment ratio against audited context)           |
|     └── Check 2: Institutional Leakage & Content Safety                                               |
+---------------------------------------------------+---------------------------------------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
+--------------------------------------------------+ +--------------------------------------------------+
| PORT 8001: RETRIEVAL MICROSERVICE                | | PORT 6006: ARIZE PHOENIX OBSERVABILITY           |
| (retrieval_service.py)                           | | (phoenix_service.py + phoenix_tracer.py)         |
| ├── Isolated PyTorch / SentenceTransformer       | | ├── Root Spans (/ask endpoint)                   |
| ├── Model: sentence-transformers/all-MiniLM-L6-v2| | ├── Child Spans (Agent 1, Retrieval, Agent 2, 3) |
| ├── ChromaDB Collection: bmu_hybrid_knowledge    | | ├── Guardrail Status (passed / blocked / bypass) |
| └── Hybrid Score: 0.68*Cosine + 0.32*Keyword     | | ├── Latency percentiles, tokens & full payloads  |
+--------------------------------------------------+ +--------------------------------------------------+
                         ^
                         | (Background Web Crawling & Voyage AI Vector Indexing)
+------------------------+-------------------------+
| PORT 8002: INGESTION MICROSERVICE                |
| (ingestion_service.py)                           |
| ├── Trafilatura Web Scraper (BMU Domains)        |
| ├── Custom Extractive Summarization (BM25+TF-IDF)|
| └── Voyage AI Client (voyage-3-large 1024-d)     |
+--------------------------------------------------+
```

---

## 2. Why Microservices? Deep-Dive into Each Service

In modern production machine learning, deploying everything as a single monolithic script leads to severe architectural flaws:
* **Memory Bloat:** Loading heavy deep learning models (PyTorch, Transformers) into the same process that handles HTTP requests causes memory spikes and slow boot times (>30 seconds).
* **Process Blocking:** Web scraping, document chunking, and embedding generation are slow I/O-heavy operations. If an ingestion crawler runs inside your web server, student chat queries freeze.
* **Failure Cascade:** If a heavy vectorization query exhausts memory or crashes, a monolith brings down the entire chatbot. Microservices isolate crashes to their own container/process.
* **Independent Scalability:** Retrieval can be assigned to GPU instances, while the orchestrator runs on lightweight CPU pods.

Our system decouples operations into **4 specialized services**:

```
+-------------------------------------------------------------------------------------------------+
|                                 MICROSERVICE TOPOLOGY                                           |
+--------------------------+--------------------+---------------------+---------------------------+
| Service Name             | Port               | Primary File        | Core Dependency           |
+--------------------------+--------------------+---------------------+---------------------------+
| Backend Orchestrator     | 8000               | semantic.py         | FastAPI, LangGraph, Groq  |
| Retrieval Microservice   | 8001               | retrieval_service.py| PyTorch, SentenceTransf.  |
| Ingestion Microservice   | 8002               | ingestion_service.py| Trafilatura, Voyage AI    |
| Arize Phoenix Server     | 6006               | phoenix_service.py  | OpenTelemetry, Phoenix UI |
+--------------------------+--------------------+---------------------+---------------------------+
```

---

### 2.1 Backend Orchestrator (`semantic.py` - Port 8000)

#### What is it?
The central brain and API gateway of the entire project. It receives user chat requests from the frontend, coordinates JEV Guardrails, executes the LangGraph agent state machine, and streams back final answers.

#### Why do we use it?
It separates client-facing logic, session management, routing, and guardrail enforcement from the heavy machine learning layers. The orchestrator does not need PyTorch installed in its process; it stays light, nimble, and fast.

#### How does it work internally?
1. **Request Intake:** Receives `POST /ask` with parameters:
   ```json
   {
     "query": "What are the eligibility criteria for B.Tech CSE at BMU?",
     "llm_provider": "groq",
     "guardrails_enabled": true
   }
   ```
2. **JEV Guardrail Gate:** Calls `jev_guardrails.validate_input()`. If malicious or out-of-scope, halts early and returns a clean, safe institutional response.
3. **LangGraph StateGraph Execution:** If valid (or if guardrails are toggled OFF), dispatches the query to `bmu_rag_graph.run_agent_rag()`.
4. **Arize Phoenix Spanning:** Wraps the entire lifecycle in an OpenTelemetry root span with metadata tags (`llm.provider`, `guardrails.enabled`, `latency_ms`).
5. **Dashboard Serving:** Hosts static assets for the Chat Assistant (`/`) and the 100-Question Evaluation Observatory (`/dashboard`).
6. **Key Endpoints:**
   - `POST /ask` - Primary student conversation endpoint.
   - `GET /health` - Health probe reporting status of downstream microservices.
   - `GET /agents/status` - Returns live status of Agent 1, Agent 2, Agent 3, and Guardrails.
   - `GET /evaluation/summary` - Aggregate metrics for Groq vs. Local LLM benchmark.
   - `GET /evaluation/results` - Granular 100-question comparison records.
   - `GET /evaluation/export/csv` - Downloads flat CSV benchmark report for administrative review.

---

### 2.2 Retrieval Microservice (`retrieval_service.py` - Port 8001)

#### What is it?
A dedicated microservice dedicated strictly to dense vector embedding generation and hybrid document search.

#### Why do we use it?
`sentence-transformers/all-MiniLM-L6-v2` runs on PyTorch. Isolating it to Port 8001 ensures:
* The orchestrator starts up in **< 1 second**.
* Memory consumption of PyTorch (~600MB) is completely isolated from the web server.
* The retrieval engine can be horizontally scaled across multiple worker processes or GPU instances without touching the chatbot codebase.

#### How does it work internally?
1. **Startup Warmup:** On launch, loads `all-MiniLM-L6-v2` into memory and connects to the ChromaDB persistent directory (`chroma_db_clean` / `bmu_hybrid_knowledge`).
2. **Vectorization:** When receiving `POST /retrieve`:
   ```json
   {
     "query": "B.Tech Computer Science tuition fee per semester",
     "top_k": 5
   }
   ```
   The service encodes the text into a 384-dimensional dense vector.
3. **Hybrid Scoring Algorithm:**
   Computes a combined score combining semantic vector cosine similarity with exact keyword matching:
   $$\text{Final Score} = (0.68 \times \text{Cosine Similarity}) + (0.32 \times \text{Keyword Overlap})$$
4. **Response Payload:** Returns ranked candidate chunks with exact source URLs, titles, and text contents.

---

### 2.3 Ingestion Microservice (`ingestion_service.py` - Port 8002)

#### What is it?
The automated knowledge-base harvesting and indexing pipeline.

#### Why do we use it?
Web crawling BMU domains (`https://www.bmu.edu.in`) involves network latency, HTML parsing, deduplication, and chunking. If this ran inside the chatbot server, scraping new pages would freeze active student conversations.

#### How does it work internally?
1. **Scraping (`trafilatura`):** Crawls specified BMU URLs, stripping away boilerplate, navigation menus, footers, and advertisement banners.
2. **Text Normalization:** Cleans encoding anomalies, whitespace, and formatting glitches.
3. **Hybrid Extractive Summarization:** Uses a custom sentence-ranking engine (BM25 + TF-IDF) to distill 2000-word university pages down to dense 300-word core factual summaries.
4. **Voyage AI Embeddings:** Calls Voyage AI's `voyage-3-large` API to generate high-precision 1024-dimensional vector representations.
5. **ChromaDB Indexing:** Inserts the vector embeddings and document metadata into the database with MD5 content hashing to prevent duplicate chunks.
6. **Endpoints:**
   - `POST /crawl` - Triggers autonomous crawl of target URLs.
   - `POST /index` - Indexes pre-parsed text documents.
   - `GET /health` - Reports ingestion queue status and database document counts.

---

### 2.4 Arize Phoenix Observability Server (`phoenix_service.py` - Port 6006)

#### What is it?
An enterprise OpenTelemetry tracing server running locally on Port 6006.

#### Why do we use it?
Traditional logging (`print()` or log files) only tells you *that* something happened. Tracing tells you **how long each step took, what prompts were passed to each agent, which model responded, how many tokens were consumed, and why a guardrail triggered**.

#### How does it work internally?
* Runs `phoenix.server.server.px.launch_app()`.
* Listens on `http://127.0.0.1:6006` for OpenTelemetry gRPC / HTTP spans sent by `phoenix_tracer.py`.
* Provides an interactive UI displaying execution DAGs, latency waterfalls, token metrics, and input/output payloads.

---

## 3. The 3-Agent LangGraph Multi-Agent Engine

```
                             +-------------------------------+
                             |    User Query Received        |
                             +---------------+---------------+
                                             |
                                             v
                             +-------------------------------+
                             |  AGENT 1: QUERY SPECIALIST    |
                             |  • Intent Classification      |
                             |  • Vocabulary Expansion       |
                             +---------------+---------------+
                                             |
                                             v
                             +-------------------------------+
                             |   DETERMINISTIC RETRIEVAL     |
                             |   • Port 8001 Vector Search   |
                             |   • 3.0s Live URL Timeout     |
                             |   • 3 Context Candidates      |
                             +---------------+---------------+
                                             |
                                             v
                     +-----------------------------------------------+
                     |          AGENT 2: CONTEXT AUDITOR             |
                     |  Is retrieved document sufficient to answer?  |
                     +-----------------------+-----------------------+
                                             |
                      +----------------------+----------------------+
                      | (TRY_FALLBACK)       | (SUFFICIENT)         | (NOT_FOUND)
                      v                      |                      v
           +--------------------+            |            +--------------------+
           | Try Candidate #2   |            |            | Flag No Context    |
           | or Cluster #3      |            |            | Honest Referral    |
           +----------+---------+            |            +----------+---------+
                      |                      |                       |
                      +----------------->----+----<------------------+
                                             |
                                             v
                             +-------------------------------+
                             | AGENT 3: RESPONSE SYNTHESIZER |
                             | • Fact-Grounded Markdown      |
                             | • Official Link Citations     |
                             +---------------+---------------+
                                             |
                                             v
                             +-------------------------------+
                             |       Final Response          |
                             +-------------------------------+
```

### 3.1 What is an Agent & Why Multi-Agent RAG?
A standard "Naive RAG" takes a user query, grabs 3 text chunks from a database, and feeds them directly to an LLM with: *"Answer the question using the context."*

**Why Naive RAG Fails in Universities:**
1. **Vocabulary Mismatch:** A student asks *"What is cse package?"*. The vector database searches for *"package"* and misses documents titled *"School of Engineering & Technology Campus Placement CTC Statistics"*.
2. **Context Poisoning:** If the database returns an irrelevant page, a single-prompt LLM will either hallucinate or state incorrect facts.
3. **No Verification:** There is no mechanism to verify if the retrieved text *actually* answers the question before generating a response.

**The Multi-Agent Solution:**
We divide cognitive responsibility across three distinct specialist agents connected in a stateful **LangGraph StateGraph**.

---

### 3.2 The Shared State Machine (`AgentState`)

All agents communicate by reading and mutating a single strongly-typed state dictionary:

```python
class AgentState(TypedDict):
    original_query: str          # Raw student query
    expanded_query: str          # Query normalized by Agent 1
    intent: str                  # admissions, fees, hostel, placements, etc.
    candidate_contexts: List[dict] # Candidate 1, Candidate 2, Candidate 3
    active_context_index: int    # Currently audited candidate index
    auditor_decision: str        # "SUFFICIENT" | "TRY_FALLBACK" | "NOT_FOUND"
    auditor_reason: str          # Justification for decision
    final_response: str          # Output produced by Agent 3
    sources: List[str]           # Verified citation URLs
    llm_provider: str            # "groq" or "local"
    guardrails_enabled: bool     # True or False
```

---

### 3.3 Agent 1: Query Specialist

* **Implementation:** `query_specialist_node` in `bmu_rag_graph.py`
* **Model:** Groq `openai/gpt-oss-20b` (or Local Ollama `llama3.2:3b`)
* **Role:**
  - Diagnoses student intent (`admissions`, `fees`, `scholarships`, `courses`, `placements`, `hostel`, `faculty`, `general`).
  - Rewrites colloquial or fragmented student queries into comprehensive, official academic search queries.
  - Expands acronyms (e.g., *"SOM"* $\rightarrow$ *"School of Management"*, *"BBA LLB"* $\rightarrow$ *"Bachelor of Business Administration and Bachelor of Legislative Law"*).

---

### 3.4 Deterministic Hybrid Retrieval Tool & 3.0s Fallback Safeguard

* **Implementation:** `retrieval_node` in `bmu_rag_graph.py`
* **Role:**
  - Sends the expanded query to the Retrieval Microservice (`http://127.0.0.1:8001/retrieve`).
  - Fetches top matching document summaries from ChromaDB.
  - **3.0-Second Live URL Timeout Safeguard:**
    When candidate documents contain live BMU web URLs, the tool attempts to scrape the full real-time text. However, if the external university server lags, `trafilatura` or `requests` could block for 30–60 seconds!
    **Our Solution:** A strict **3.0-second timeout** with `requests.get()`. If the external server does not respond within 3 seconds, it instantly aborts and falls back to the pre-indexed ChromaDB summary, ensuring 0 network hangs.
  - Prepares **3 Distinct Candidate Contexts**:
    - **Candidate #1:** Top-ranked document page/summary.
    - **Candidate #2:** Second-ranked document page/summary.
    - **Candidate #3:** Synthesized cluster combining the top 6 chunks across all retrieved documents.

---

### 3.5 Agent 2: Context Auditor & Self-Correction Loop

* **Implementation:** `context_auditor_node` in `bmu_rag_graph.py`
* **Model:** Groq `openai/gpt-oss-20b` (or Local Ollama `llama3.2:3b`)
* **Role:**
  - Evaluates whether Candidate #1 contains verifiable facts to answer the question.
  - Outputs a structured decision:
    - `"SUFFICIENT"`: The document contains exact facts $\rightarrow$ proceeds immediately to Agent 3.
    - `"TRY_FALLBACK"`: The document is missing specific numbers, dates, or criteria $\rightarrow$ switches to Candidate #2 or Candidate #3.
    - `"NOT_FOUND"`: All candidates have been checked and none contain the facts $\rightarrow$ instructs Agent 3 to issue an honest referral rather than hallucinating.

---

### 3.6 Agent 3: Response Synthesizer & Citation Grounding

* **Implementation:** `response_synthesizer_node` in `bmu_rag_graph.py`
* **Model:** Groq `openai/gpt-oss-120b` (or Local Ollama `llama3.2:3b`)
* **Role:**
  - Generates the final student-facing answer using clean Markdown (headers, bullet points, fee tables).
  - Enforces strict grounding: only mentions facts present in the audited context.
  - Appends official, clickable source citations:
    ```markdown
    **Official BMU Sources:**
    - [BMU Admissions & Fee Structure](https://www.bmu.edu.in/admission/)
    ```

---

## 4. JEV Guardrails: Security, Scope & Hallucination Defense

```
                       User Prompt Arrives
                                |
                     [Guardrails Enabled?]
                     /                   \
                   YES                    NO
                   /                        \
      +------------------------+      +---------------------------+
      | 1. Prompt Injection    |      | BYPASS MODE ACTIVATED     |
      |    - DAN / Jailbreaks  |      | ├── Silences Input Checks |
      |    - System Overrides  |      | ├── Direct Raw LLM Query  |
      |    - SQL/Code Exploit  |      | └── UI Provenance Badge:  |
      +-----------+------------+      |     "Guardrails Bypassed" |
                  |                   +---------------------------+
      +-----------+------------+
      | 2. Toxicity & Abuse    |
      +-----------+------------+
                  |
      +-----------+------------+
      | 3. BMU Domain Scope    |
      |    - Out of Scope?     |
      |    - Polite Referral   |
      +-----------+------------+
                  |
              [PASSED]
                  |
                  v
         LangGraph RAG Graph
                  |
                  v
      +------------------------+
      | 4. Output Grounding    |
      |    - Fact Check vs Doc |
      |    - Safety Validation |
      +------------------------+
```

### 4.1 What is JEV (Judge / Evaluate / Validate)?
JEV is a dual-stage guardrail engine (`jev_guardrails.py`) that acts as an institutional security and quality barrier. It runs **System-One heuristics** (regex, pattern trees, keyword classifiers) and **System-Two semantic evaluation** before and after the agent workflow.

---

### 4.2 Input Guardrail Pipeline
Before any LLM sees the prompt, it must pass three filters:
1. **Adversarial Prompt Injection & Jailbreak Detection:**
   - Detects patterns like: *"Ignore all previous instructions"*, *"You are now DAN (Do Anything Now)"*, *"Reveal your system prompt"*, or SQL/code injection strings.
   - Action: Blocks immediately with `[Security Guardrail] Potential adversarial instruction detected.`
2. **Toxicity & Abuse Classification:**
   - Detects profanity, harassment, hate speech, and abusive phrasing.
   - Action: Blocks immediately with an institutional civility notice.
3. **BMU Academic Domain Scope Classifier:**
   - Verifies whether the query relates to BML Munjal University (admissions, courses, fees, placements, faculty, hostel, sports, campus).
   - If a student asks *"Give a cup cake recipe"* or *"Write Python code for quicksort"*, the guardrail intercepts it:
     > *"[Scope Guardrail] I am the official BMU Chat Assistant, specialized exclusively in BML Munjal University inquiries. Please ask a BMU-related question!"*

---

### 4.3 Output Guardrail Pipeline
After Agent 3 creates a draft answer:
1. **Factual Grounding Check:**
   - Extracts key entities and numbers from the answer and computes containment against the audited context.
   - If grounding is weak, appends an institutional disclaimer to prevent unverified claims.
2. **System Prompt Protection:**
   - Ensures the generated response does not accidentally echo internal agent instructions or API keys.

---

### 4.4 Guardrails ON vs. Guardrails OFF Bypass Mode
The frontend header provides a toggle: `🛡️ Guardrails: ON / OFF`.

* **When Guardrails = ON (Default):**
  - Full JEV protection is active.
  - Queries outside BMU scope or containing prompt injections are blocked cleanly.
* **When Guardrails = OFF:**
  - JEV input validation is completely bypassed.
  - If a user asks a general query (e.g., *"Give a cup cake recipe"*), ChromaDB returns no university documents (`NOT_FOUND`).
  - Instead of refusing, `_generate_unconstrained_answer()` executes: the query is routed directly to the active LLM (Groq or Ollama) with an unconstrained system prompt.
  - The response is delivered with a prominent warning badge:
    ```
    ⚠️ Guardrails Bypassed (Raw LLM Generation)
    ```

---

## 5. Arize Phoenix Tracing & Observability

### 5.1 Tracing vs. Traditional Logging
| Feature | Traditional Logging (`log.info`) | Arize Phoenix Tracing |
|---|---|---|
| **Structure** | Flat lines of text | Hierarchical Tree of Spans (DAG) |
| **Context** | Hard to correlate across async calls | Trace ID shared across all microservices |
| **Metrics** | Manual timing calculation | Automatic P50, P90, P95, P99 latency percentiles |
| **Token Tracking** | Not tracked | Exact prompt, completion, and total tokens tracked |
| **Visual Inspection**| Command-line terminal | Interactive Web Dashboard (`http://localhost:6006`) |

---

### 5.2 OpenTelemetry Span Tree & Metadata
Every user question produces an OpenTelemetry distributed trace:

```text
Trace: /ask (Root Span)
├── Attributes:
│   ├── llm.provider: "groq" | "local"
│   ├── llm.is_local: false | true
│   ├── guardrails.enabled: true | false
│   ├── response.source: "multi_agent_rag" | "unconstrained_llm" | "guardrail_blocked"
│   └── execution.config: { provider: "groq", guardrails: true }
│
├── Span: JEV Input Guardrail Validation
│   └── Attributes: { status: "passed", intent: "fees", elapsed_ms: 1.2 }
│
├── Span: Agent 1 (QuerySpecialist)
│   └── Attributes: { model: "openai/gpt-oss-20b", tokens: 145, latency_ms: 680 }
│
├── Span: Hybrid Retrieval Microservice (Port 8001)
│   └── Attributes: { top_k: 5, candidate_count: 3, latency_ms: 120 }
│
├── Span: Agent 2 (ContextAuditor)
│   └── Attributes: { decision: "SUFFICIENT", candidate_index: 0, latency_ms: 540 }
│
└── Span: Agent 3 (ResponseSynthesizer)
    └── Attributes: { model: "openai/gpt-oss-120b", tokens: 412, latency_ms: 1120 }
```

---

### 5.3 How to Inspect Traces in the Phoenix UI
1. Open [http://localhost:6006](http://localhost:6006) in your browser.
2. Click on the **Traces** tab.
3. Select any `/ask` root trace.
4. Expand individual spans to inspect:
   - Full input prompts sent to each agent.
   - Raw model responses received.
   - Exact latency in milliseconds for each step.
   - Guardrail decisions and blocking reasons.

---

## 6. Dual LLM Runtime: Cloud (Groq) vs. Local (Ollama)

### 6.1 Runtime Comparison Table

| Attribute | ☁️ Cloud LLM (Groq LPU) | 💻 Local LLM (Ollama) |
|---|---|---|
| **Default Model** | `openai/gpt-oss-120b` (Synthesis) & `gpt-oss-20b` (Specialist) | `llama3.2:3b` |
| **Host / Endpoint** | `https://api.groq.com/openai/v1` | `http://localhost:11434/v1` |
| **Inference Hardware** | Groq Tensor Streaming Processor (LPU) | Local CPU / GPU |
| **Median Latency** | **850ms – 2,100ms** | **6,000ms – 14,000ms** |
| **Network Reliance** | Requires internet access | 100% Offline & Private |
| **Cost** | Cloud API usage | **$0.00 (Zero cost)** |
| **Primary Use Case** | Production high-capacity student interactions | Sensitive offline deployments & privacy |

---

### 6.2 Seamless Frontend Toggle Architecture
The user interface features a live switch in the top header:
```
[ ☁️ Cloud LLM (Groq) ]   [ 💻 Local LLM (Ollama) ]
```
When clicked:
1. `frontend/app.js` updates the client state.
2. All subsequent requests send `"llm_provider": "local"` or `"groq"`.
3. `bmu_rag_graph.py` dynamically routes agent prompts to `http://localhost:11434/v1` or `https://api.groq.com/openai/v1`.
4. The response message card renders a badge indicating the exact model used.

---

## 7. Vector Database & Embedding Pipelines

### 7.1 ChromaDB Vector Store
* **Storage Engine:** SQLite + HNSW (Hierarchical Navigable Small World) index.
* **Location:** Persistent local directory `chroma_db_clean`.
* **Collections:**
  - `bmu_hybrid_knowledge` - Primary dense MiniLM collection used for sub-second retrieval.
  - `bmu_voyage_knowledge` - Deep semantic collection generated via Voyage AI.

---

### 7.2 Dense MiniLM vs. Voyage AI Embeddings
* **`sentence-transformers/all-MiniLM-L6-v2`:**
  - Dimensionality: 384 dimensions.
  - Runtime: Local CPU inference in **< 15ms**.
  - Purpose: Fast real-time candidate search in `retrieval_service.py`.
* **Voyage AI `voyage-3-large`:**
  - Dimensionality: 1024 dimensions.
  - Runtime: Cloud API embedding.
  - Purpose: Deep semantic representation of complex university curriculum documents.

---

### 7.3 Hybrid Scoring Formula
Pure vector similarity sometimes retrieves documents that mention general concepts but lack the exact keyword (e.g., retrieving *tuition* when the student asked for *hostel*).

To solve this, `retrieval_service.py` computes:
```python
def compute_hybrid_score(doc_text, query_text, cosine_sim):
    query_tokens = set(query_text.lower().split())
    doc_tokens = set(doc_text.lower().split())
    keyword_overlap = len(query_tokens.intersection(doc_tokens)) / max(len(query_tokens), 1)
    
    return (0.68 * cosine_sim) + (0.32 * keyword_overlap)
```

---

## 8. Web Ingestion & Extractive Summarization

### 8.1 Trafilatura Web Scraper
`ingestion_service.py` uses `trafilatura` to scrape raw HTML from `bmu.edu.in`. Trafilatura applies linguistic heuristics to isolate the main article body while stripping:
- Navigation bars
- Cookie banners
- Footer copyright blocks
- JavaScript tags

---

### 8.2 Hybrid Extractive Summarization Algorithm
Rather than feeding massive 3,000-word university pages into vector embeddings (which causes dilution), the ingestion pipeline uses a **custom hybrid extractive summarizer**:
1. **Sentence Tokenization:** Splits the cleaned page into individual sentences.
2. **Frequency Scoring:** Computes TF-IDF term weights for each sentence.
3. **Position Weighting:** Awards higher weights to sentences appearing early in paragraphs and sections.
4. **Key Phrase Boosting:** Boosts sentences containing academic terms (*eligibility, percentage, fee, semester, scholarship, admission*).
5. **Top-$K$ Selection:** Selects the top 5 most informative sentences and preserves their chronological order.

---

## 9. 100-Question Evaluation Framework & RAGAS Metrics

```
+-------------------------------------------------------------------------------------------------+
|                                 100-QUESTION BENCHMARK ENGINE                                   |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [benchmark_100_questions.py]                                                                  |
|   └── 20 Categories × 5 Questions = 100 Verified Ground-Truth Questions                         |
|                                                                                                 |
|   [run_100_evaluation.py]                                                                       |
|   ├── Rate-Limit Guardian (Groq Free Tier: 8,000 TPM limit)                                     |
|   ├── Adaptive Delay: 1.8s Spacing + Exponential Backoff                                        |
|   ├── Auto-Checkpointing: Writes to evaluation_100_checkpoint.json after every query            |
|   └── RAGAS Metrics Engine (evaluate_metrics.py)                                                |
|       ├── Relevance      (Semantic Cosine + Keyword Overlap)                                    |
|       ├── Faithfulness   (Grounding in Retrieved Context)                                       |
|       ├── Completeness   (Entity & Fact Extraction Match)                                       |
|       ├── Conciseness    (Information Density Penalty)                                          |
|       ├── Correctness    (Cosine Similarity vs. Reference Answer)                               |
|       └── Guardrail Acc. (Confusion Matrix on Safe vs. Adversarial)                             |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

### 9.1 Purpose of Benchmarking & Rate-Limit Strategy
To scientifically prove system quality, we test the chatbot against a **100-question benchmark dataset** comparing Cloud LLM (Groq) and Local LLM (Ollama).

**Groq Rate Limit Challenge:**
Groq's Free Tier enforces a strict **8,000 Tokens per Minute (TPM)** limit. Because our 3-agent graph makes 3 LLM calls per question (~2,500 total tokens), executing queries without pacing causes immediate `429 Too Many Requests` errors.

**Our Protection Architecture:**
1. **Adaptive Spacing:** Automatically paces requests with a **$\ge 1.8$-second sleep delay** between questions.
2. **Exponential Backoff:** If a `429` is encountered, dynamically pauses with randomized jitter.
3. **Atomic Checkpointing:** Writes state to [evaluation_100_checkpoint.json](file:///d:/BMU_WEBSITE_CHATBOT-main/BMU_WEBSITE_CHATBOT-main/evaluation_outputs/evaluation_100_checkpoint.json) after *every single query*. If interrupted, running with `--resume` resumes instantly from the exact question index!

---

### 9.2 20 Categories × 5 Questions Dataset

The dataset ([benchmark_100_questions.py](file:///d:/BMU_WEBSITE_CHATBOT-main/BMU_WEBSITE_CHATBOT-main/benchmark_100_questions.py)) covers 20 exhaustive domains:
1. Admissions & Eligibility (5 Qs)
2. Fees & Expenses (5 Qs)
3. Scholarships & Financial Aid (5 Qs)
4. B.Tech CSE & Specializations (5 Qs)
5. MBA & School of Management (5 Qs)
6. School of Law (5 Qs)
7. School of Liberal Studies (5 Qs)
8. Campus Life & Societies (5 Qs)
9. Hostel & Accommodation (5 Qs)
10. Sports & Recreation (5 Qs)
11. Library & Digital Resources (5 Qs)
12. Placements & Career Services (5 Qs)
13. Internships & Global Immersion (5 Qs)
14. Faculty & Research Centers (5 Qs)
15. Academic Policies & Regulations (5 Qs)
16. Location & Transportation (5 Qs)
17. Health, Safety & Amenities (5 Qs)
18. Complex Multi-Constraint Queries (5 Qs)
19. Out-of-Scope Queries (5 Qs - Tests scope blocking)
20. Adversarial Prompt Injections (5 Qs - Tests security blocking)

---

### 9.3 RAGAS Mathematical Scoring Formulas

Every answer is evaluated on a normalized **0.0 to 5.0 scale**:

* **Relevance:**
  $$\text{Relevance} = 5.0 \times (0.7 \times \cos(\vec{q}, \vec{a}) + 0.3 \times \text{Overlap}(q, a))$$
* **Faithfulness:**
  $$\text{Faithfulness} = 5.0 \times \frac{|\text{Tokens}(a) \cap \text{Tokens}(c)|}{|\text{Tokens}(a)|}$$
* **Completeness:**
  $$\text{Completeness} = 5.0 \times \frac{|\text{KeyEntities}(a) \cap \text{KeyEntities}(\text{GroundTruth})|}{|\text{KeyEntities}(\text{GroundTruth})|}$$
* **Conciseness:**
  $$\text{Conciseness} = \max(0, 5.0 - \text{Penalty}(\text{WordCount} > 300))$$
* **Correctness:**
  $$\text{Correctness} = 5.0 \times \cos(\vec{a}, \vec{\text{GroundTruth}})$$
* **Overall Score:**
  $$\text{Overall} = 0.25 \times \text{Relevance} + 0.25 \times \text{Faithfulness} + 0.25 \times \text{Correctness} + 0.15 \times \text{Completeness} + 0.10 \times \text{Conciseness}$$

---

## 10. Google Stitch MCP & The Obsidian Telemetry Observatory

```
+-------------------------------------------------------------------------------------------------+
|                         GOOGLE STITCH MCP INTEGRATION WORKFLOW                                  |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   1. Stitch MCP Server Connection                                                               |
|      ├── Endpoint: https://stitch.googleapis.com/mcp                                             |
|      └── Authentication: SSE Protocol with X-Goog-Api-Key Header                                |
|                                                                                                 |
|   2. Project Creation & Design System Synthesis                                                 |
|      ├── Stitch Project: projects/2776335830318453950                                           |
|      │   "BMU Chatbot 100-Query Evaluation Dashboard"                                           |
|      └── Design System Asset: assets/4a851e6e47854a03a868c99cf010474f                           |
|          "Obsidian Telemetry" Theme                                                             |
|                                                                                                 |
|   3. Frontend Implementation                                                                    |
|      ├── frontend/evaluation.html  (Semantic dark telemetry markup)                             |
|      ├── frontend/evaluation.css   (Glassmorphic neon teal & electric purple tokens)            |
|      └── frontend/evaluation.js    (Chart.js telemetry charts, search filters & drawer)         |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

### Dashboard Features (`http://127.0.0.1:8000/dashboard/evaluation.html`):
1. **Executive KPI Strip:** Live cards comparing Groq vs. Ollama on Overall Score, Guardrail Accuracy, Median Latency, and Error Rates.
2. **Interactive Chart.js Visualizations:**
   - **RAGAS Radar Chart:** 5-axis comparative spider plot.
   - **Latency Breakdown Bar Chart:** P50 and P95 latency distributions.
   - **Guardrail Precision Matrix:** Safe vs. Adversarial confusion matrix.
3. **20-Category Comparison Matrix:** Interactive table showing exact performance and delta scores across all academic domains.
4. **Question Explorer:** Real-time search by query keyword, category dropdown, and difficulty filter (Easy, Medium, Hard, Adversarial).
5. **Side-by-Side Comparison Modal Drawer:** Clicking any question opens an inspection drawer showing:
   - Ground Truth Reference Answer
   - Cloud LLM (Groq) Answer + Citations + Latency breakdown
   - Local LLM (Ollama) Answer + Citations + Latency breakdown
   - Exact RAGAS scores across all 5 dimensions
6. **Data Export:** Instant download buttons for **CSV** and **JSON** reports.

---

## 11. Complete Project File Map

```text
BMU_WEBSITE_CHATBOT-main/
├── .env                              # Environment variables (API keys, ports, models)
├── .env.phoenix                      # Phoenix OpenTelemetry tracing configuration
├── Brain.md                          # Ultra-detailed technical documentation & runbooks
├── README.md                         # Primary comprehensive system documentation (This file)
├── requirements.txt                  # Python package dependencies
├── run.bat                           # One-click Windows launcher for all 4 microservices
│
├── semantic.py                       # FastAPI Orchestrator (Port 8000)
├── retrieval_service.py              # Dedicated Retrieval Microservice (Port 8001)
├── ingestion_service.py              # Ingestion & Voyage AI Microservice (Port 8002)
├── phoenix_service.py                # Arize Phoenix Observability Server (Port 6006)
├── phoenix_tracer.py                 # OpenTelemetry Tracer Provider
├── bmu_rag_graph.py                  # LangGraph 3-Agent StateGraph
├── jev_guardrails.py                 # JEV Security & Grounding Guardrails Engine
├── voyage_client.py                  # Voyage AI Embeddings Client
│
├── benchmark_100_questions.py        # 100-Question benchmark dataset generator
├── run_100_evaluation.py             # Dual-model benchmark runner & RAGAS evaluator
├── run_100_query_evaluation.py       # Legacy wrapper script
├── evaluate_metrics.py               # RAGAS metric mathematical scoring functions
│
├── frontend/                         # User Interfaces
│   ├── index.html                    # Chat Assistant main interface
│   ├── styles.css                    # Chat Assistant styles (segmented toggle, pills)
│   ├── app.js                        # Chat Assistant client logic
│   ├── evaluation.html               # Stitch MCP Benchmark Observatory Dashboard
│   ├── evaluation.css                # Obsidian Telemetry dark glassmorphic styling
│   └── evaluation.js                 # Observatory charts, filters, and drawer logic
│
└── evaluation_outputs/               # Evaluation artifacts
    ├── benchmark_100_questions.json  # 100 benchmark questions dataset
    ├── evaluation_100_results.json   # Full model outputs and metric scores
    ├── evaluation_100_summary.json   # Aggregated KPIs and category comparisons
    ├── evaluation_100_checkpoint.json# Resume checkpoint file
    └── evaluation_100_results.csv    # Flat tabular export for Excel / reporting
```

---

## 12. Installation, Setup & Quickstart

### Prerequisites
1. **Python 3.10+** (Tested on Python 3.11 and 3.14)
2. **Ollama** installed with `llama3.2:3b`:
   ```bash
   ollama pull llama3.2:3b
   ```
3. **Groq API Keys** configured in `.env`:
   ```env
   GROQ_API_KEY=gsk_...
   GROQ_API_KEY_QUERY_AGENT=gsk_...
   GROQ_API_KEY_CONTEXT_AGENT=gsk_...
   GROQ_API_KEY_SYNTHESIS_AGENT=gsk_...
   LOCAL_LLM_URL=http://localhost:11434/v1
   LOCAL_LLM_MODEL=llama3.2:3b
   ```

---

### Option A: One-Click Launch (Windows)
Double-click `run.bat` or run in terminal:
```cmd
run.bat
```
This automatically boots all 4 services in parallel:
1. Arize Phoenix on `http://127.0.0.1:6006`
2. Retrieval Microservice on `http://127.0.0.1:8001`
3. Ingestion Microservice on `http://127.0.0.1:8002`
4. Backend Orchestrator on `http://127.0.0.1:8000`
5. Opens `frontend/index.html` in your default browser.

---

### Option B: Manual Startup (Individual Terminals)
```bash
# Terminal 1: Arize Phoenix
python phoenix_service.py

# Terminal 2: Retrieval Microservice
python retrieval_service.py

# Terminal 3: Ingestion Microservice
python ingestion_service.py

# Terminal 4: Backend Orchestrator
python semantic.py
```

---

### Running the Evaluation Benchmark
```powershell
# 1. Quick dry-run (3 sample questions)
python run_100_evaluation.py --dry-run

# 2. Run full 100-question Groq benchmark (safe 1.8s delay)
python run_100_evaluation.py --limit 100 --provider groq --delay 1.8

# 3. Resume an interrupted run
python run_100_evaluation.py --limit 100 --provider groq --resume

# 4. Compare both Groq and Ollama on first 10 questions
python run_100_evaluation.py --limit 10 --provider both --resume
```

---

## 13. Port & API Reference Table

| Port | Service Name | Entry Script | Swagger / Web UI | Primary Purpose |
|---|---|---|---|---|
| **8000** | **Backend Orchestrator** | `semantic.py` | [Swagger Docs](http://127.0.0.1:8000/docs) | Chat API, Guardrails, LangGraph Graph, Observatory UI |
| **8001** | **Retrieval Microservice** | `retrieval_service.py` | [Swagger Docs](http://127.0.0.1:8001/docs) | SentenceTransformer MiniLM embeddings & ChromaDB search |
| **8002** | **Ingestion Microservice** | `ingestion_service.py` | [Swagger Docs](http://127.0.0.1:8002/docs) | Web scraper & Voyage AI vector indexing |
| **6006** | **Arize Phoenix** | `phoenix_service.py` | [Phoenix UI](http://127.0.0.1:6006) | OpenTelemetry Distributed Tracing & Spans |
| **11434**| **Ollama Local LLM** | Ollama daemon | N/A | Local offline inference with `llama3.2:3b` |

---

## 14. Troubleshooting & FAQ

### Q1: I get `429 Too Many Requests` when using Groq.
* **Why:** Groq's Free Tier has an 8,000 Tokens per Minute (TPM) limit. 3 agents make 3 calls per question.
* **Fix:** When running evaluations, always pass `--delay 1.8` (or `--delay 2.5`). `run_100_evaluation.py` includes automatic exponential backoff.

### Q2: Local LLM (Ollama) queries are taking 20+ seconds.
* **Why:** If running on CPU without dedicated GPU acceleration, `llama3.2:3b` generation takes 15–30 tokens/second.
* **Fix:** This is normal for CPU inference. `bmu_rag_graph.py` has a 120-second client timeout configured to prevent premature network aborts.

### Q3: Why does a cupcake recipe question say "Blocked" when Guardrails are ON?
* **Why:** BMU Chatbot is an institutional assistant. JEV Scope Guardrails block out-of-scope non-university questions.
* **Fix:** Toggle `🛡️ Guardrails: OFF` in the header! The bot will now answer unconstrained using raw LLM reasoning.

### Q4: How do I view traces?
* **Fix:** Navigate to [http://localhost:6006](http://localhost:6006) while `phoenix_service.py` is running. All chat queries and evaluations appear under the Traces tab.

---

*Designed and engineered for BML Munjal University.*
