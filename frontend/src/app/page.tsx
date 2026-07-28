"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import Header from "@/components/Header";
import ImageViewer from "@/components/ImageViewer";
import DiagnosticPanel from "@/components/DiagnosticPanel";
import { predictImage } from "@/lib/api";
import { ImageControls, PredictionResponse } from "@/types";

const DEFAULT_CONTROLS: ImageControls = {
  brightness: 100,
  contrast: 100,
  invert: false,
  heatmapOpacity: 0,
  zoom: 1,
};

export default function Home() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [controls, setControls] = useState<ImageControls>(DEFAULT_CONTROLS);
  const [originalBase64, setOriginalBase64] = useState("");

  const handleFileSelect = useCallback(async (file: File, base64: string) => {
    setOriginalBase64(base64);
    setError(null);
    setResult(null);
    setControls((c) => ({ ...c, heatmapOpacity: 0 }));
    setLoading(true);

    try {
      const prediction = await predictImage(file);
      setResult(prediction);
      // Auto-enable 40% Grad-CAM overlay after result arrives
      setTimeout(() => {
        setControls((c) => ({ ...c, heatmapOpacity: 40 }));
      }, 600);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleControlChange = useCallback(
    (key: keyof ImageControls, value: number | boolean) => {
      setControls((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  return (
    <div className="min-h-screen grid-bg flex flex-col">
      <Header />

      {/* Error toast */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-auto mt-4 w-full max-w-[1600px] px-6"
        >
          <div className="rounded-xl border border-red-700/50 bg-red-950/40 px-4 py-3 text-sm text-red-300 flex items-center gap-2">
            <span className="text-red-400">⚠</span>
            <strong>API Error:</strong>&nbsp;{error}
            <span className="text-xs text-red-400/70 ml-1">
              — Ensure the FastAPI backend is running on port 8000.
            </span>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-200 text-xs"
            >
              Dismiss
            </button>
          </div>
        </motion.div>
      )}

      {/* Main workstation layout */}
      <main className="flex-1 mx-auto w-full max-w-[1600px] px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_420px] gap-6 h-full">
          {/* Left panel — Image Viewer */}
          <motion.section
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col gap-2"
          >
            <SectionLabel label="Interactive Medical Image Viewer" />
            <div className="flex-1">
              <ImageViewer
                onFileSelect={handleFileSelect}
                heatmapBase64={result?.heatmap_base64 ?? null}
                controls={controls}
                onControlChange={handleControlChange}
              />
            </div>
          </motion.section>

          {/* Right panel — Diagnostics */}
          <motion.section
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="flex flex-col gap-2"
          >
            <SectionLabel label="Real-Time Diagnostic Intelligence" />
            <div className="flex-1 overflow-y-auto pr-0.5">
              <DiagnosticPanel
                result={result}
                loading={loading}
                originalBase64={originalBase64}
              />
            </div>
          </motion.section>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/60 py-3 text-center text-[11px] text-slate-600">
        Pediatric Pneumonia AI Diagnostic Workstation ·{" "}
        <span className="text-slate-500">Software by Abdullah Ishaq</span> ·
        ResNet-50 · ROC-AUC 0.978 · Sensitivity 96.9%
      </footer>
    </div>
  );
}

function SectionLabel({ label }: { label: string }) {
  return (
    <p className="text-[11px] text-slate-500 uppercase tracking-widest font-semibold px-1 mb-1">
      {label}
    </p>
  );
}
