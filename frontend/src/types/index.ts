// TypeScript interfaces for API payloads
export interface PredictionResponse {
  prediction: "PNEUMONIA" | "NORMAL";
  confidence: number;
  prob_normal: number;
  prob_pneumonia: number;
  threshold: number;
  heatmap_base64: string;
}

export interface HealthResponse {
  status: string;
  model: string;
  device: string;
  threshold: number;
  version: string;
}

export interface PDFRequest {
  prediction: string;
  confidence: number;
  prob_normal: number;
  prob_pneumonia: number;
  threshold: number;
  heatmap_base64: string;
  original_base64: string;
}

export interface ImageControls {
  brightness: number;
  contrast: number;
  invert: boolean;
  heatmapOpacity: number;
  zoom: number;
}
