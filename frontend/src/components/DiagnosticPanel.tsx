"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Info,
  SlidersHorizontal,
} from "lucide-react";
import { PredictionResponse } from "@/types";
import PdfExportBtn from "./PdfExportBtn";
import AnalyticsDrawer from "./AnalyticsDrawer";

interface Props {
  result: PredictionResponse | null;
  loading: boolean;
  originalBase64: string;
}

export default function DiagnosticPanel({ result, loading, originalBase64 }: Props) {
  const [threshold, setThreshold] = useState(0.50);
  const [effectivePrediction, setEffectivePrediction] = useState<"PNEUMONIA" | "NORMAL" | null>(null);

  // Sync threshold with result when a new prediction arrives
  useEffect(() => {
    if (result) {
      setThreshold(result.threshold);
    }
  }, [result]);

  // Re-evaluate prediction when threshold slider changes
  useEffect(() => {
    if (!result) return;
    setEffectivePrediction(
      result.prob_pneumonia >= threshold ? "PNEUMONIA" : "NORMAL"
    );
  }, [result, threshold]);

  if (loading) {
    return (
      <div className="flex flex-col gap-4 h-full">
        <SkeletonCard />
        <SkeletonCard height="h-28" />
        <SkeletonCard height="h-40" />
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex flex-col gap-4 h-full">
        {/* Empty state */}
        <div className="glass-card p-8 flex flex-col items-center justify-center gap-4 text-center flex-1 min-h-[200px]">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-800/80 border border-slate-700/50">
            <Info className="h-7 w-7 text-slate-500" />
          </div>
          <div>
            <p className="text-slate-300 font-semibold">No Analysis Yet</p>
            <p className="text-xs text-slate-500 mt-1">
              Upload a chest X-ray to run the AI diagnostic pipeline
            </p>
          </div>
        </div>
        <AnalyticsDrawer />
      </div>
    );
  }

  const isPneumonia = effectivePrediction === "PNEUMONIA";
  const confidence = isPneumonia ? result.prob_pneumonia : result.prob_normal;

  return (
    <div className="flex flex-col gap-4">
      {/* Primary result card */}
      <AnimatePresence mode="wait">
        <motion.div
          key={effectivePrediction}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className={`glass-card p-5 border-l-4 ${
            isPneumonia
              ? "border-l-red-500"
              : "border-l-emerald-500"
          }`}
        >
          <div className="flex items-start justify-between gap-3 mb-3">
            <div className="flex items-center gap-3">
              <div
                className={`flex h-10 w-10 items-center justify-center rounded-xl shrink-0 ${
                  isPneumonia
                    ? "bg-red-950/60 border border-red-700/50 pneumonia-pulse"
                    : "bg-emerald-950/60 border border-emerald-700/50"
                }`}
              >
                {isPneumonia ? (
                  <AlertTriangle className="h-5 w-5 text-red-400" />
                ) : (
                  <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                )}
              </div>
              <div>
                <p
                  className={`text-lg font-bold tracking-tight ${
                    isPneumonia ? "text-red-400" : "text-emerald-400"
                  }`}
                >
                  {isPneumonia ? "PNEUMONIA DETECTED" : "NORMAL LUNG PARENCHYMA"}
                </p>
                <p className="text-xs text-slate-400 mt-0.5">
                  {isPneumonia
                    ? "Focal pulmonary opacity detected"
                    : "Clear lung parenchyma — no abnormality"}
                </p>
              </div>
            </div>
            <span
              className={`shrink-0 rounded-full px-3 py-1 text-sm font-bold ${
                isPneumonia
                  ? "bg-red-950/60 text-red-300 border border-red-700/40"
                  : "bg-emerald-950/60 text-emerald-300 border border-emerald-700/40"
              }`}
            >
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>

          {/* Probability bars */}
          <div className="flex flex-col gap-2 mt-3">
            <ProbBar
              label="NORMAL"
              value={result.prob_normal}
              color="bg-blue-500"
            />
            <ProbBar
              label="PNEUMONIA"
              value={result.prob_pneumonia}
              color="bg-red-500"
            />
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Threshold tuner */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card p-5"
      >
        <div className="flex items-center gap-2 mb-3">
          <SlidersHorizontal className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200">
            Decision Threshold Tuner
          </h3>
        </div>

        <div className="flex items-center gap-3 mb-2">
          <input
            id="threshold-slider"
            type="range"
            min={0.1}
            max={0.99}
            step={0.01}
            value={threshold}
            className="red-thumb flex-1"
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
          <span className="font-mono text-sm font-bold text-white w-12 text-right">
            {threshold.toFixed(3)}
          </span>
        </div>

        {/* Tradeoff indicator */}
        <div className="grid grid-cols-2 gap-2 mt-3">
          <TradeoffCell
            label="Sensitivity ↑"
            value={threshold < 0.5 ? "Very High" : threshold < 0.8 ? "High" : threshold < 0.93 ? "Moderate" : "Conservative"}
            color={threshold < 0.5 ? "text-emerald-400" : threshold < 0.93 ? "text-amber-400" : "text-red-400"}
          />
          <TradeoffCell
            label="Specificity ↑"
            value={threshold > 0.95 ? "Very High" : threshold > 0.8 ? "High" : threshold > 0.5 ? "Moderate" : "Low"}
            color={threshold > 0.95 ? "text-emerald-400" : threshold > 0.8 ? "text-amber-400" : "text-red-400"}
          />
        </div>
        <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">
          Default 0.50 = Optimal balanced decision threshold.
          Lower threshold → higher sensitivity (fewer missed cases).
          Higher → higher specificity (fewer false alarms).
        </p>
      </motion.div>

      {/* Clinical findings */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass-card p-5"
      >
        <h3 className="text-sm font-semibold text-slate-200 mb-3">
          Clinical Findings
        </h3>
        <ul className="flex flex-col gap-2">
          {(isPneumonia
            ? [
                "Focal pulmonary opacity identified in Grad-CAM activation regions",
                "Pattern consistent with lobar or bronchopneumonia consolidation",
                "Recommend clinical correlation with CXR and laboratory findings",
                "Consider antibiotic therapy per local paediatric guidelines",
                "Differential: bacterial, viral, or aspiration pneumonia",
              ]
            : [
                "No focal opacities or consolidation detected by AI attention map",
                "Lung parenchyma appears radiologically clear",
                "No pleural effusion or pneumothorax pattern identified",
                "Continue clinical observation and symptomatic management",
                "Negative result does not exclude very early-stage disease",
              ]
          ).map((finding, i) => (
            <motion.li
              key={i}
              initial={{ x: -10, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.25 + i * 0.06 }}
              className={`flex items-start gap-2 text-xs text-slate-300 leading-relaxed`}
            >
              <span
                className={`mt-0.5 h-1.5 w-1.5 rounded-full shrink-0 ${
                  isPneumonia ? "bg-red-500" : "bg-emerald-500"
                }`}
              />
              {finding}
            </motion.li>
          ))}
        </ul>
      </motion.div>

      {/* PDF export */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <PdfExportBtn result={result} originalBase64={originalBase64} />
      </motion.div>

      {/* Analytics drawer */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <AnalyticsDrawer />
      </motion.div>

      {/* Disclaimer */}
      <p className="text-[10px] text-slate-600 leading-relaxed text-center px-2">
        ⚠ Research prototype. Not for clinical use. All findings must be validated by
        a qualified radiologist or clinician.
      </p>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ProbBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400 w-24 shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>
      <span className="text-xs font-mono text-slate-300 w-12 text-right">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function TradeoffCell({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="rounded-lg bg-slate-900/60 border border-slate-700/40 p-2.5 text-center">
      <p className={`text-sm font-bold ${color}`}>{value}</p>
      <p className="text-[10px] text-slate-500 mt-0.5">{label}</p>
    </div>
  );
}

function SkeletonCard({ height = "h-48" }: { height?: string }) {
  return (
    <div className={`glass-card p-5 ${height} animate-pulse`}>
      <div className="h-3 w-2/3 bg-slate-700/60 rounded mb-3" />
      <div className="h-3 w-1/2 bg-slate-700/40 rounded mb-2" />
      <div className="h-3 w-3/4 bg-slate-700/40 rounded" />
    </div>
  );
}
