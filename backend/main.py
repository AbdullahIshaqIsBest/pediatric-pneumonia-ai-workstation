"""
main.py — FastAPI Backend
=========================
Endpoints:
  GET  /api/health          — liveness / model status
  POST /api/predict         — image upload → classification + Grad-CAM
  POST /api/generate-pdf    — diagnostic payload → downloadable PDF

Run locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Docker / Render:
    CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
import io

from backend.model_service import load_model, predict, OPTIMAL_THRESHOLD
from backend.pdf_service import generate_pdf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pediatric Pneumonia AI Diagnostic API",
    description=(
        "Production-grade REST API for classifying pediatric chest X-rays "
        "using a fine-tuned ResNet-50 with Grad-CAM attention visualisation. "
        "Software by Abdullah Ishaq."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Vercel frontend + localhost dev
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://*.vercel.app",
    os.environ.get("FRONTEND_URL", ""),
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production to specific origins list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup — load model weights once
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    weights_candidates = [
        Path("saved_models/best_resnet50_pneumonia.pth"),
        Path("../saved_models/best_resnet50_pneumonia.pth"),
        Path("saved_models/best_model.pth"),
        Path("outputs/best_pneumonia_model.pth"),
        Path("../outputs/best_pneumonia_model.pth"),
        Path(os.environ.get("MODEL_WEIGHTS_PATH", "saved_models/best_resnet50_pneumonia.pth")),
    ]
    weights_path: Path | None = None
    for candidate in weights_candidates:
        if candidate.exists():
            weights_path = candidate
            break

    if weights_path is None:
        logger.warning(
            "No model weights found — starting in DEMO mode with random weights. "
            "Set MODEL_WEIGHTS_PATH env variable to point to best_pneumonia_model.pth."
        )
    load_model(weights_path)
    logger.info("API startup complete.")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    prediction:     str   = Field(..., description="PNEUMONIA or NORMAL")
    confidence:     float = Field(..., description="Confidence for winning class (0-1)")
    prob_normal:    float = Field(..., description="P(NORMAL)")
    prob_pneumonia: float = Field(..., description="P(PNEUMONIA)")
    threshold:      float = Field(..., description="Decision threshold used")
    heatmap_base64: str   = Field(..., description="Base64-encoded Grad-CAM PNG overlay")


class PDFRequest(BaseModel):
    prediction:     str
    confidence:     float
    prob_normal:    float
    prob_pneumonia: float
    threshold:      float  = OPTIMAL_THRESHOLD
    heatmap_base64: str    = ""
    original_base64: str   = ""


class HealthResponse(BaseModel):
    status:    str
    model:     str
    device:    str
    threshold: float
    version:   str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness / readiness probe."""
    return HealthResponse(
        status="ok",
        model="ResNet-50 | Target Layer: layer4",
        device="cpu",
        threshold=OPTIMAL_THRESHOLD,
        version="1.0.0",
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def api_predict(file: UploadFile = File(...)):
    """
    Accept a chest X-ray image (PNG / JPG / JPEG) and return:
    - Classification result (PNEUMONIA / NORMAL)
    - Class probabilities
    - Base64-encoded Grad-CAM heatmap overlay
    """
    # Validate MIME type
    allowed = {"image/png", "image/jpeg", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: PNG, JPG, JPEG.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    try:
        result = predict(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return PredictionResponse(**result)


@app.post("/api/generate-pdf", tags=["Report"])
async def api_generate_pdf(payload: PDFRequest):
    """
    Generate and stream a downloadable clinical PDF diagnostic report.
    """
    try:
        pdf_bytes = generate_pdf(payload.model_dump())
    except Exception as exc:
        logger.exception("PDF generation error")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=pneumonia_diagnostic_report.pdf"
        },
    )
