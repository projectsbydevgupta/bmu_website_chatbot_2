"""
BMU Chatbot - Unified Production Multi-Service Orchestrator
===========================================================
Boots and manages all three services in a single process supervisor:
  1. Arize Phoenix Observability Server  (Port 6006)
  2. BGE 1,024-d Retrieval Microservice  (Port 8001)
  3. Semantic Orchestrator & Frontend    (Port 8000 or $PORT)

Compatible with Windows, Linux, AWS EC2, and Render containers.
"""

import os
import signal
import subprocess
import sys
import time

PYTHON_CMD = sys.executable

processes: list[subprocess.Popen] = []


def cleanup(signum=None, frame=None):
    print("\n[Orchestrator] Shutting down all BMU Chatbot services...")
    for p in processes:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                p.kill()
    print("[Orchestrator] All services stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def start_service(cmd: list[str], name: str, port: int) -> subprocess.Popen:
    print(f"[Orchestrator] Launching {name} on port {port}...")
    p = subprocess.Popen(
        cmd,
        env=os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    processes.append(p)
    return p


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    port = os.environ.get("PORT", "8000")

    print("=" * 65)
    print("      BMU CHATBOT - UNIFIED PRODUCTION STACK LAUNCHER       ")
    print("=" * 65)
    print(f"Working Directory: {root_dir}")
    print(f"Main Port:        {port}")
    print(f"Python:           {PYTHON_CMD}")
    print("=" * 65)

    # 1. Start Arize Phoenix Observability (Port 6006)
    disable_phoenix = os.environ.get("DISABLE_LOCAL_PHOENIX", "false").lower() == "true"
    if not disable_phoenix:
        start_service(
            [PYTHON_CMD, os.path.join(root_dir, "phoenix_service.py")],
            "Arize Phoenix Observability Server",
            6006,
        )
        time.sleep(2)
    else:
        print("[Orchestrator] Local Phoenix disabled (using cloud or external collector).")

    # 2. Start Retrieval Microservice (Port 8001)
    start_service(
        [PYTHON_CMD, os.path.join(root_dir, "retrieval_service.py")],
        "BGE-1024 Retrieval Microservice",
        8001,
    )
    time.sleep(2)

    # 3. Start Main Semantic Orchestrator & Frontend Dashboard (Port 8000 / $PORT)
    print(f"[Orchestrator] Launching Semantic Orchestrator on host 0.0.0.0 port {port}...")
    semantic_proc = subprocess.Popen(
        [
            PYTHON_CMD,
            "-m",
            "uvicorn",
            "semantic:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        env=os.environ.copy(),
    )
    processes.append(semantic_proc)

    print("\n" + "=" * 65)
    print("  ✅ All BMU Chatbot services are LIVE and connected!")
    print(f"  👉 Chatbot & Evaluation Dashboard: http://localhost:{port}")
    print(f"  👉 BGE 1,024-d Microservice:       http://localhost:8001")
    print(f"  👉 Arize Phoenix Observability:    http://localhost:6006")
    print("=" * 65 + "\n")

    # Monitor processes
    while True:
        for p in processes:
            ret = p.poll()
            if ret is not None:
                print(f"[Orchestrator] Service PID {p.pid} exited with code {ret}.")
                cleanup()
        time.sleep(1)


if __name__ == "__main__":
    main()

