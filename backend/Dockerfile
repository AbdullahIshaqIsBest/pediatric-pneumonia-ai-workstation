# ─── Stage 1: build ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system libs needed by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt requirements.txt

# CPU-only PyTorch keeps the image small (~900 MB vs ~4 GB for CUDA)
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.0+cpu torchvision==0.18.0+cpu \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# ─── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Model weights are mounted/downloaded at runtime via MODEL_WEIGHTS_PATH env var
ENV MODEL_WEIGHTS_PATH=outputs/best_pneumonia_model.pth
ENV PYTHONPATH=/app

EXPOSE 8000

# Use shell form so dynamic $PORT environment variable from Railway is evaluated properly
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
