const API_BASE_URL = "http://127.0.0.1:8000";

const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const suggestionButtons = document.querySelectorAll(".suggestion-btn");

// Toggle controls
const llmCloudBtn = document.getElementById("llmCloudBtn");
const llmLocalBtn = document.getElementById("llmLocalBtn");
const guardrailsToggle = document.getElementById("guardrailsToggle");
const guardrailsStatusText = document.getElementById("guardrailsStatusText");

let selectedLlm = "groq";
let guardrailsEnabled = true;

if (llmCloudBtn && llmLocalBtn) {
  llmCloudBtn.addEventListener("click", () => {
    selectedLlm = "groq";
    llmCloudBtn.classList.add("active");
    llmCloudBtn.setAttribute("aria-checked", "true");
    llmLocalBtn.classList.remove("active");
    llmLocalBtn.setAttribute("aria-checked", "false");
  });

  llmLocalBtn.addEventListener("click", () => {
    selectedLlm = "local";
    llmLocalBtn.classList.add("active");
    llmLocalBtn.setAttribute("aria-checked", "true");
    llmCloudBtn.classList.remove("active");
    llmCloudBtn.setAttribute("aria-checked", "false");
  });
}

if (guardrailsToggle && guardrailsStatusText) {
  guardrailsToggle.addEventListener("change", () => {
    guardrailsEnabled = guardrailsToggle.checked;
    if (guardrailsEnabled) {
      guardrailsStatusText.textContent = "ON";
      guardrailsStatusText.className = "switch-status-text active";
    } else {
      guardrailsStatusText.textContent = "OFF";
      guardrailsStatusText.className = "switch-status-text inactive";
    }
  });
}

function addMessage(
  role,
  text,
  sources = [],
  guardrail = null,
  agentTrace = [],
  llmProvider = null,
  guardrailsActive = null
) {
  const node = document.createElement("article");
  node.className = `msg ${role}`;

  // Render Source Provenance Bar (for bot responses from backend)
  if (role === "bot" && llmProvider) {
    const sourceBar = document.createElement("div");
    sourceBar.className = "msg-source-bar";

    const llmPill = document.createElement("span");
    if (llmProvider === "local") {
      llmPill.className = "source-pill llm-local";
      llmPill.innerHTML = "💻 Local LLM (Ollama)";
    } else {
      llmPill.className = "source-pill llm-cloud";
      llmPill.innerHTML = "☁️ Cloud LLM (Groq)";
    }
    sourceBar.appendChild(llmPill);

    const guardPill = document.createElement("span");
    if (guardrailsActive) {
      guardPill.className = "source-pill guard-active";
      guardPill.innerHTML = "🛡️ Guardrails: ON";
    } else {
      guardPill.className = "source-pill guard-off";
      guardPill.innerHTML = "⚠️ Guardrails: OFF";
    }
    sourceBar.appendChild(guardPill);

    node.appendChild(sourceBar);
  }

  if (role === "bot" && guardrail) {
    if (guardrail.status === "blocked") {
      node.classList.add("msg-blocked");
    }

    const badge = document.createElement("div");
    badge.className = `guardrail-badge ${guardrail.status}`;
    if (guardrail.status === "blocked") {
      badge.textContent = `🛡️ JEV Guardrail: Blocked (${guardrail.reason || "Policy Violation"})`;
      node.appendChild(badge);
    } else if (guardrail.status === "flagged") {
      badge.textContent = "🛡️ JEV Output Guardrail: Disclaimer Appended";
      node.appendChild(badge);
    } else if (guardrail.action === "retrieval_gated") {
      badge.textContent = "🛡️ Relevance Gated: Low Knowledge Base Match";
      node.appendChild(badge);
    } else if (guardrail.status === "bypassed") {
      badge.textContent = "⚠️ Guardrails Bypassed (Raw LLM Generation)";
      node.appendChild(badge);
    }
  }

  const textNode = document.createElement("div");
  textNode.className = "msg-text";
  textNode.textContent = text;
  node.appendChild(textNode);

  if (role === "bot" && Array.isArray(agentTrace) && agentTrace.length > 0) {
    const traceCard = document.createElement("details");
    traceCard.className = "agent-trace-card";
    const summary = document.createElement("summary");
    summary.innerHTML = `⚡ <strong>LangGraph Multi-Agent Workflow</strong> (${agentTrace.length} steps)`;
    traceCard.appendChild(summary);

    const list = document.createElement("ul");
    list.className = "agent-trace-list";
    agentTrace.forEach((item) => {
      const li = document.createElement("li");
      const providerLabel = item.details?.provider ? `[${item.details.provider.toUpperCase()}] ` : "";
      li.innerHTML = `<strong>${item.agent}</strong> ${providerLabel}— <span class="trace-latency">${item.latency_ms}ms</span>`;
      list.appendChild(li);
    });
    traceCard.appendChild(list);
    node.appendChild(traceCard);
  }

  if (role === "bot" && Array.isArray(sources) && sources.length > 0) {
    const metaCard = document.createElement("div");
    metaCard.className = "meta-card";

    const header = document.createElement("strong");
    header.textContent = "References";
    metaCard.appendChild(header);

    sources.forEach((item, idx) => {
      const source = document.createElement("div");
      source.className = "source-item";
      const score = Number(item.score ?? 0).toFixed(3);
      source.innerHTML = `${idx + 1}. <a href="${item.url}" target="_blank" rel="noopener noreferrer">${item.url}</a> (score ${score})`;
      metaCard.appendChild(source);
    });

    node.appendChild(metaCard);
  }

  chatMessages.appendChild(node);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function askBackend(query) {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      llm_provider: selectedLlm,
      enable_guardrails: guardrailsEnabled,
    }),
  });

  if (!response.ok) {
    let details = "Request failed";
    try {
      const body = await response.json();
      details = body.detail || details;
    } catch (error) {
      details = response.statusText || details;
    }
    throw new Error(details);
  }

  return response.json();
}

async function submitQuery(query) {
  const userQuery = query.trim();
  if (!userQuery) {
    return;
  }

  addMessage("user", userQuery);
  queryInput.value = "";
  sendBtn.disabled = true;

  try {
    const data = await askBackend(userQuery);
    addMessage(
      "bot",
      data.answer || "No answer returned",
      data.top_results || [],
      data.guardrail || null,
      data.agent_trace || [],
      data.llm_provider || selectedLlm,
      typeof data.guardrails_enabled === "boolean" ? data.guardrails_enabled : guardrailsEnabled
    );
  } catch (error) {
    addMessage("bot", `Unable to process request: ${error.message}`);
  } finally {
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitQuery(queryInput.value);
});

suggestionButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const q = button.dataset.query || "";
    queryInput.value = q;
    await submitQuery(q);
  });
});

addMessage("bot", "Hi! Ask me anything about BMU.");
