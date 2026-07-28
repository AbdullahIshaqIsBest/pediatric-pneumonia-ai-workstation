FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY backend/requirements.txt requirements.txt

# Install CPU PyTorch and lightweight headless requirements without any apt-get debian packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.3.0+cpu torchvision==0.18.0+cpu \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy proprietary application code
COPY . .

# Environment variables for PyTorch weights and module path
ENV MODEL_WEIGHTS_PATH=outputs/best_pneumonia_model.pth
ENV PYTHONPATH=/app

EXPOSE 8000

# Start server using Railway's dynamic $PORT
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
