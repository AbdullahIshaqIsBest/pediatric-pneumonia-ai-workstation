# 🏥 Pediatric Pneumonia AI Diagnostic Workstation

> **Software by Abdullah Ishaq**  
> **Copyright (c) 2026 Abdullah Ishaq. All rights reserved.**  
> Full-stack AI web application for classifying pediatric chest X-rays using a fine-tuned ResNet-50 with Grad-CAM attention visualization.

---

## 🏗️ Architecture

```
medical_ai_research/
├── backend/                  # FastAPI + PyTorch CPU + ReportLab
│   ├── main.py               # REST API (/api/health, /api/predict, /api/generate-pdf)
│   ├── model_service.py      # In-memory ResNet-50 + Grad-CAM inference
│   ├── pdf_service.py        # ReportLab clinical PDF generator
│   ├── requirements.txt      # CPU-only PyTorch dependencies
│   └── Dockerfile            # Multi-stage build for Render/Koyeb/HF Spaces
└── frontend/                 # Next.js 14 App Router + TypeScript + Tailwind CSS
    ├── src/
    │   ├── app/              # layout.tsx, page.tsx, globals.css
    │   ├── components/       # Header, ImageViewer, DiagnosticPanel, AnalyticsDrawer, PdfExportBtn
    │   ├── lib/api.ts        # Typed fetch client
    │   └── types/index.ts    # TypeScript interfaces
    └── vercel.json           # Vercel deployment config
```

---

## 🚀 Local Development

### 1. Start the FastAPI Backend

```bash
# From the medical_ai_research/ root directory
pip install fastapi uvicorn python-multipart reportlab pillow numpy opencv-python-headless

# Install CPU-only PyTorch (lightweight)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Run the API server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will start at **http://localhost:8000**
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

> 📌 **Model Weights**: Place `best_pneumonia_model.pth` from your Colab training run into `outputs/`. The API auto-discovers it at startup.

### 2. Start the Next.js Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## ☁️ Free Cloud Deployment

### Frontend → Vercel (Free)

1. Push the `frontend/` folder to a GitHub repo.
2. Import the repo at **vercel.com/new**.
3. Set environment variable:
   ```
   NEXT_PUBLIC_API_URL = https://your-api.onrender.com
   ```
4. Deploy — done!

### Backend → Render (Free Tier)

1. Push the entire `medical_ai_research/` repo to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Set:
   - **Build Command**: `pip install -r backend/requirements.txt --index-url https://download.pytorch.org/whl/cpu`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variable**: `MODEL_WEIGHTS_PATH=outputs/best_pneumonia_model.pth`
4. Upload `best_pneumonia_model.pth` to the `outputs/` folder in the repo.

### Backend → Docker (Any cloud)

```bash
cd medical_ai_research/
docker build -f backend/Dockerfile -t pneumonia-api .
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs pneumonia-api
```

---

## 🔑 Key Features

| Feature | Details |
|---|---|
| **Backbone** | ResNet-50, fine-tuned on Kaggle Chest X-Ray Dataset |
| **Grad-CAM** | Target layer: `layer4` — real-time attention heatmap overlay |
| **Decision Threshold** | Default 0.933 (Youden's J from ROC-AUC analysis) |
| **ROC-AUC** | 0.978 |
| **Sensitivity** | 96.9% |
| **PDF Report** | ReportLab clinical summary with embedded X-ray and CAM images |
| **Interactive Controls** | Zoom, pan, brightness, contrast, invert, CAM opacity |
| **Analytics Drawer** | Embedded confusion matrix, ROC curve, training curves |

---

## 📊 Adding Evaluation Figures

Copy your IEEE-ready figures into the frontend public folder:

```bash
cp outputs/confusion_matrix.png  frontend/public/figures/
cp outputs/roc_curve.png         frontend/public/figures/
cp outputs/training_curves.png   frontend/public/figures/
```

They will automatically appear in the **Model Analytics** modal drawer.

---

## 🚨 Proprietary Research Notice

**Copyright (c) 2026 Abdullah Ishaq. All rights reserved.**

This web workstation and REST API are proprietary research property of **Abdullah Ishaq**. Unauthorized copying, modification, or public redistribution is strictly prohibited without prior written permission. Once the associated clinical research is published in a peer-reviewed journal, an open-source public release will be made available.

*Software Architecture & Research Pipeline by **Abdullah Ishaq** — Pediatric Pneumonia AI Research*
