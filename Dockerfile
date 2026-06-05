# ── Folio Backend ──────────────────────────────────────────
# Python FastAPI backend for Ethos persona management
# Build:  docker build -t ethos-folio -f Dockerfile .
# Run:    docker run --env-file .env -p 8001:8001 ethos-folio

FROM python:3.12-slim

WORKDIR /app

# Install system deps for opencv-python-headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

# Copy data
COPY data/ data/    

# Expose the API port
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
