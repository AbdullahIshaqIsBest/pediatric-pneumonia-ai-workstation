"use client";

import { motion } from "framer-motion";
import { Download, FileText, Loader2 } from "lucide-react";
import { useState } from "react";
import { generatePDF } from "@/lib/api";
import { PDFRequest, PredictionResponse } from "@/types";

interface Props {
  result: PredictionResponse;
  originalBase64: string;
}

export default function PdfExportBtn({ result, originalBase64 }: Props) {
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const handleExport = async () => {
    setLoading(true);
    setDone(false);
    try {
      const payload: PDFRequest = {
        prediction: result.prediction,
        confidence: result.confidence,
        prob_normal: result.prob_normal,
        prob_pneumonia: result.prob_pneumonia,
        threshold: result.threshold,
        heatmap_base64: result.heatmap_base64,
        original_base64: originalBase64,
      };

      const blob = await generatePDF(payload);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "pneumonia_diagnostic_report.pdf";
      a.click();
      URL.revokeObjectURL(url);
      setDone(true);
      setTimeout(() => setDone(false), 3000);
    } catch (err) {
      console.error("PDF export failed:", err);
      alert("PDF generation failed. Ensure the backend API is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.97 }}
      onClick={handleExport}
      disabled={loading}
      className={`flex w-full items-center justify-center gap-2.5 rounded-xl px-4 py-3 text-sm font-semibold transition-all ${
        done
          ? "bg-emerald-600/80 border border-emerald-500/50 text-white"
          : "bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white shadow-lg shadow-cyan-900/30"
      } disabled:opacity-60 disabled:cursor-not-allowed`}
    >
      {loading ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          Generating PDF…
        </>
      ) : done ? (
        <>
          <Download className="h-4 w-4" />
          Downloaded!
        </>
      ) : (
        <>
          <FileText className="h-4 w-4" />
          Export Clinical Summary (PDF)
        </>
      )}
    </motion.button>
  );
}
