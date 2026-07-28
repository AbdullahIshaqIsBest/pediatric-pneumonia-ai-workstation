"""
app.py
======
Hugging Face Spaces (Gradio SDK — 100% Free Tier) Entry Point.

Mounts our proprietary FastAPI REST backend onto a clean Gradio interface,
enabling 100% free hosting with 16 GB of RAM without requiring paid Docker tiers!
"""
import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import gradio as gr
import uvicorn
from backend.main import app as fastapi_app

# Set default checkpoint path for HF Spaces if not explicitly set
if "MODEL_WEIGHTS_PATH" not in os.environ:
    os.environ["MODEL_WEIGHTS_PATH"] = "outputs/best_pneumonia_model.pth"

# ─── Minimal UI for HF Spaces Landing Page ─────────────────────────────────
with gr.Blocks(title="Pediatric Pneumonia AI — REST API") as demo:
    gr.Markdown("# 🏥 Pediatric Pneumonia AI Workstation — REST API Backend")
    gr.Markdown(
        """
        > **Proprietary Research Software by Abdullah Ishaq**  
        > **Copyright (c) 2026 Abdullah Ishaq. All rights reserved.**
        
        This Hugging Face Space hosts the high-performance PyTorch + FastAPI REST API backend for our clinical workstation.
        """
    )
    with gr.Row():
        gr.Markdown(
            """
            ### 🔗 Active REST Endpoints:
            - `GET /api/health` — Check server status and model device
            - `POST /api/predict` — Perform ResNet-50 inference + Grad-CAM generation
            - `POST /api/generate-pdf` — Generate ReportLab clinical diagnostic summary
            """
        )

# Mount Gradio onto our FastAPI app at the root "/" path
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    # Hugging Face Spaces routes web traffic to port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
