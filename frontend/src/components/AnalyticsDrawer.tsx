"use client";

import { motion } from "framer-motion";
import {
  ChevronDown,
  BarChart2,
  X,
} from "lucide-react";
import { useState } from "react";
import Image from "next/image";

const FIGURES = [
  {
    id: "confusion_matrix",
    title: "Confusion Matrix",
    desc: "Classification performance breakdown across NORMAL / PNEUMONIA classes.",
    // In production, serve from /outputs/ via a static asset host or Next.js public folder
    src: "/figures/confusion_matrix.png",
  },
  {
    id: "roc_curve",
    title: "ROC Curve",
    desc: "Receiver Operating Characteristic curve — AUC = 0.978.",
    src: "/figures/roc_curve.png",
  },
  {
    id: "training_curves",
    title: "Training Curves",
    desc: "Stage-1 (frozen backbone) and Stage-2 (full fine-tune) loss & accuracy curves.",
    src: "/figures/training_curves.png",
  },
];

export default function AnalyticsDrawer() {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);

  return (
    <>
      {/* Trigger button */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={() => setOpen(true)}
        className="flex w-full items-center justify-between gap-2 rounded-xl border border-indigo-700/40 bg-indigo-950/30 px-4 py-3 text-sm font-medium text-indigo-300 hover:bg-indigo-950/50 transition-colors"
      >
        <span className="flex items-center gap-2">
          <BarChart2 className="h-4 w-4" />
          Model Analytics & Evaluation Figures
        </span>
        <ChevronDown className="h-4 w-4 opacity-60" />
      </motion.button>

      {/* Modal overlay */}
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          onClick={() => setOpen(false)}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

          {/* Modal panel */}
          <motion.div
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            onClick={(e) => e.stopPropagation()}
            className="relative z-10 w-full max-w-4xl glass-card-glow p-6"
          >
            {/* Header */}
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-slate-100">
                  Model Evaluation Analytics
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  ResNet-50 · Test Set · IEEE Publication Figures
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="rounded-lg p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Tab strip */}
            <div className="flex gap-1 rounded-lg bg-slate-900/60 p-1 mb-5">
              {FIGURES.map((fig, i) => (
                <button
                  key={fig.id}
                  onClick={() => setActive(i)}
                  className={`flex-1 rounded-md px-3 py-2 text-xs font-semibold transition-all ${
                    active === i
                      ? "bg-cyan-600/80 text-white shadow"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {fig.title}
                </button>
              ))}
            </div>

            {/* Figure display */}
            <div className="rounded-xl bg-slate-900/60 p-4 flex flex-col items-center gap-3 min-h-[360px]">
              <div className="relative w-full max-w-2xl aspect-[4/3] bg-slate-800/40 rounded-lg overflow-hidden flex items-center justify-center">
                <img
                  src={FIGURES[active].src}
                  alt={FIGURES[active].title}
                  className="max-h-full max-w-full object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
                <span className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 text-xs select-none pointer-events-none gap-1">
                  <BarChart2 className="h-8 w-8 opacity-20" />
                  <span className="opacity-40">
                    Copy outputs/*.png into frontend/public/figures/
                  </span>
                </span>
              </div>
              <p className="text-xs text-slate-400 text-center max-w-md">
                {FIGURES[active].desc}
              </p>
            </div>

            {/* Stats row */}
            <div className="mt-4 grid grid-cols-4 gap-3">
              {[
                { label: "Accuracy", value: "96.2%" },
                { label: "Sensitivity", value: "96.9%" },
                { label: "Specificity", value: "94.1%" },
                { label: "ROC-AUC", value: "0.978" },
              ].map((s) => (
                <div
                  key={s.label}
                  className="rounded-lg bg-slate-900/60 border border-slate-700/40 p-3 text-center"
                >
                  <p className="text-lg font-bold text-cyan-400">{s.value}</p>
                  <p className="text-[10px] text-slate-400 mt-0.5 uppercase tracking-wide">
                    {s.label}
                  </p>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </>
  );
}
