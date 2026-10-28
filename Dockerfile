# Multi-stage lightweight Python Dockerfile for BMU Chatbot Stack
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project source code
COPY . .

# Expose ports: 8000 (Semantic / Frontend), 8001 (Retrieval Microservice), 6006 (Arize Phoenix)
EXPOSE 8000 8001 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch all 3 services via orchestrator
CMD ["python", "start_services.py"]

