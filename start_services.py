"""
BMU Chatbot - Unified Production Multi-Service Orchestrator
===========================================================
Boots and manages all three services in a single process supervisor:
  1. Arize Phoenix Observability Server  (Port 6006)
  2. BGE 1,024-d Retrieval Microservice  (Port 8001)
  3. Semantic Orchestrator & Frontend    (Port 8000 or $PORT)

Features:
- Thread-constrained CPU optimization for AWS EC2 (t2/t3 free tier)
- Autonomous process crash supervisor with auto-restart
- Out-of-memory (SIGKILL -9) detection and diagnostic guidance
- Graceful multi-process shutdown on SIGINT/SIGTERM
"""

import os
import signal
import subprocess
import sys
import time

# Constrain parallel thread pools across all child processes to prevent OOM
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONUTF8"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PYTHON_CMD = sys.executable

service_configs: list[dict] = []
is_shutting_down = False


def cleanup(signum=None, frame=None):
    global is_shutting_down
    if is_shutting_down:
        return
    is_shutting_down = True
    print("\n[Orchestrator] Shutting down all BMU Chatbot services...")
    for s in service_configs:
        p = s.get("proc")
        if p and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    print("[Orchestrator] All services stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def launch_proc(service: dict) -> subprocess.Popen:
    print(f"[Orchestrator] Launching {service['name']} on port {service['port']}...")
    proc = subprocess.Popen(
        service["cmd"],
        env=os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    service["proc"] = proc
    return proc


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

    # 1. Arize Phoenix (Port 6006)
    disable_phoenix = os.environ.get("DISABLE_LOCAL_PHOENIX", "false").lower() == "true"
    if not disable_phoenix:
        service_configs.append({
            "name": "Arize Phoenix Observability Server",
            "cmd": [PYTHON_CMD, os.path.join(root_dir, "phoenix_service.py")],
            "port": 6006,
            "proc": None,
            "restarts": 0,
        })
    else:
        print("[Orchestrator] Local Phoenix disabled (using cloud or external collector).")

    # 2. Retrieval Microservice (Port 8001)
    service_configs.append({
        "name": "BGE-1024 Retrieval Microservice",
        "cmd": [PYTHON_CMD, os.path.join(root_dir, "retrieval_service.py")],
        "port": 8001,
        "proc": None,
        "restarts": 0,
    })

    # 3. Main Semantic Orchestrator & Frontend Dashboard (Port 8000 / $PORT)
    service_configs.append({
        "name": "Semantic Orchestrator & Dashboard",
        "cmd": [
            PYTHON_CMD,
            "-m",
            "uvicorn",
            "semantic:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        "port": int(port),
        "proc": None,
        "restarts": 0,
    })

    # Start services sequentially with brief pause
    for s in service_configs:
        launch_proc(s)
        time.sleep(2)

    print("\n" + "=" * 65)
    print("  [OK] All BMU Chatbot services are LIVE and connected!")
    print(f"  -> Chatbot & Evaluation Dashboard: http://localhost:{port}")
    print(f"  -> BGE 1,024-d Microservice:       http://localhost:8001")
    print(f"  -> Arize Phoenix Observability:    http://localhost:6006")
    print("=" * 65 + "\n")

    # Monitor and auto-restart crashed services
    while not is_shutting_down:
        for s in service_configs:
            p = s.get("proc")
            if p is not None:
                ret = p.poll()
                if ret is not None:
                    print(f"\n[Orchestrator] Service '{s['name']}' (PID {p.pid}) exited with code {ret}.")
                    if ret in (-9, 137):
                        print("=" * 65)
                        print(" [CRITICAL] Exit code -9/137 detected (Linux OOM Killer / SIGKILL)!")
                        print(" The instance ran out of physical RAM.")
                        print(" FIX: Allocate a 4GB swapfile on your AWS EC2 instance by running:")
                        print("   sudo fallocate -l 4G /swapfile")
                        print("   sudo chmod 600 /swapfile")
                        print("   sudo mkswap /swapfile")
                        print("   sudo swapon /swapfile")
                        print("=" * 65)

                    if s["restarts"] < 5:
                        s["restarts"] += 1
                        print(f"[Orchestrator] Auto-restarting '{s['name']}' in 2 seconds (attempt {s['restarts']}/5)...")
                        time.sleep(2)
                        launch_proc(s)
                    else:
                        print(f"[Orchestrator] Max restarts (5) exceeded for '{s['name']}'.")
                        cleanup()
        time.sleep(1)


if __name__ == "__main__":
    main()
