# Dockerfile for the UPIQAL FastAPI inference backend.
# Use this on a CPU-capable container host (Fly.io / Render / Railway).
# Vercel handles the static frontend separately; see vercel.json.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# System packages for Pillow/OpenCV-style decoding (libjpeg, libpng, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libpng16-16 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch / torchvision from the CPU-only wheel index first — this keeps
# the final image ~800MB instead of ~4GB when CUDA wheels would be pulled.
COPY requirements.txt /app/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" "torchvision>=0.17" \
 && pip install -r /app/requirements.txt

# Project sources (upiqal package, CLI, and web app).
COPY upiqal/ /app/upiqal/
COPY web/    /app/web/
COPY weights/ /app/weights/
COPY upiqal_cli.py /app/upiqal_cli.py
COPY pyproject.toml /app/pyproject.toml

EXPOSE 8000

# Healthcheck — the server exposes /healthz once FastAPI is ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/healthz || exit 1

CMD ["sh", "-c", "uvicorn web.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
