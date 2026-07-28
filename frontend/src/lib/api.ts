// API client connecting to NEXT_PUBLIC_API_URL
import { HealthResponse, PDFRequest, PredictionResponse } from "@/types";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/api/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("API health check failed");
  return res.json();
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}

export async function generatePDF(payload: PDFRequest): Promise<Blob> {
  const res = await fetch(`${API_URL}/api/generate-pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("PDF generation failed");
  return res.blob();
}
