"""
BMU Chatbot - Keep-Alive Heartbeat Cron Service
===============================================
Pings the BMU Chatbot server every 5 minutes (300 seconds) to ensure
the server remains continuously awake and prevents cloud hosts (e.g., Render Free Tier)
from spinning down into sleep mode.

Usage:
  py cron_keep_alive.py
  py cron_keep_alive.py --url https://your-server.onrender.com --interval 300
"""

import argparse
import datetime
import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()


def run_keep_alive(target_url: str, interval_seconds: int = 300):
    health_url = target_url.rstrip("/") + "/health"
    print("=" * 65)
    print("   BMU CHATBOT - 5-MINUTE KEEP-ALIVE HEARTBEAT SERVICE   ")
    print("=" * 65)
    print(f"Target URL:        {target_url}")
    print(f"Health Endpoint:   {health_url}")
    print(f"Ping Interval:     {interval_seconds}s ({interval_seconds // 60} minutes)")
    print(f"Started at:        {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print("Press Ctrl+C to stop.\n")

    pings_sent = 0
    successful_pings = 0

    while True:
        pings_sent += 1
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            t0 = time.time()
            resp = requests.get(health_url, timeout=15)
            latency_ms = round((time.time() - t0) * 1000, 1)

            if resp.status_code == 200:
                successful_pings += 1
                status_icon = "🟢"
                status_note = f"HTTP 200 OK ({latency_ms} ms) -> Server ACTIVE"
            else:
                status_icon = "🟡"
                status_note = f"HTTP {resp.status_code} ({latency_ms} ms)"

            print(f"[{now_str}] {status_icon} Ping #{pings_sent}: {status_note}")

        except requests.exceptions.RequestException as e:
            print(f"[{now_str}] 🔴 Ping #{pings_sent} FAILED: {e}")

        # Sleep until next heartbeat interval
        try:
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nKeep-alive service stopped by user.")
            print(f"Summary: {successful_pings}/{pings_sent} successful heartbeats.")
            sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMU Chatbot Keep-Alive Daemon")
    default_target = os.getenv("SERVER_URL", "http://127.0.0.1:8000")
    parser.add_argument("--url", default=default_target, help="Base URL of BMU chatbot server")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds (default: 300 = 5 min)")
    args = parser.parse_args()

    run_keep_alive(args.url, args.interval)

