"""
Phoenix AI Observability & Tracing Server
=========================================
Runs Arize Phoenix locally on port 6006 for capturing LLM spans, evaluations,
and multi-agent LangGraph workflow traces.
"""

import os
import sys

# Configure environment defaults
os.environ["PYTHONUTF8"] = "1"
port = os.environ.get("PHOENIX_PORT", "6006")
host = os.environ.get("PHOENIX_HOST", "0.0.0.0")

if __name__ == "__main__":
    print(f"Starting Arize Phoenix Server on http://{host}:{port}...")
    from phoenix.server.main import main
    sys.argv = ["phoenix", "serve", "--host", host, "--port", str(port)]
    main()
