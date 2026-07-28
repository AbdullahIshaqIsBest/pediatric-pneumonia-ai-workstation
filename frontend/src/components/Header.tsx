"use client";

import { motion } from "framer-motion";
import {
  Activity,
  Brain,
  CheckCircle2,
  Layers,
  Target,
  XCircle,
} from "lucide-react";
import { useEffect, useState } from "react";
import { checkHealth } from "@/lib/api";
import { HealthResponse } from "@/types";

export default function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setLoading(false));
  }, []);

  const isConnected = health?.status === "ok";

  return (
    <header className="relative z-20 border-b border-indigo-950/60 bg-[#070b14]/90 backdrop-blur-xl">
      {/* top accent line */}
      <div className="h-[2px] w-full bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-60" />

      <div className="mx-auto max-w-[1600px] px-6 py-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        {/* Left — branding */}
        <div className="flex items-start gap-4">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500/20 to-indigo-600/20 border border-cyan-500/30 shrink-0"
          >
            <Brain className="h-6 w-6 text-cyan-400" />
          </motion.div>

          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h1 className="text-xl font-bold tracking-tight text-slate-100 leading-tight">
              Pediatric Pneumonia AI Diagnostic Workstation
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Software by{" "}
              <span className="text-cyan-400 font-medium">Abdullah Ishaq</span>
            </p>
          </motion.div>
        </div>

        {/* Right — status badges */}
        <motion.div
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="flex flex-wrap items-center gap-2"
        >
          {/* API status */}
          <StatusBadge
            loading={loading}
            ok={isConnected}
            labelOk="API Connected"
            labelFail="API Offline"
          />

          {/* Model info pills */}
          {[
            { icon: Layers, label: "ResNet-50" },
            { icon: Target, label: "layer4" },
            { icon: Activity, label: "ROC-AUC 0.978" },
          ].map(({ icon: Icon, label }) => (
            <span
              key={label}
              className="flex items-center gap-1.5 rounded-full bg-slate-800/60 border border-slate-700/60 px-3 py-1 text-xs text-slate-300 font-medium"
            >
              <Icon className="h-3 w-3 text-indigo-400" />
              {label}
            </span>
          ))}
        </motion.div>
      </div>
    </header>
  );
}

function StatusBadge({
  loading,
  ok,
  labelOk,
  labelFail,
}: {
  loading: boolean;
  ok: boolean;
  labelOk: string;
  labelFail: string;
}) {
  if (loading) {
    return (
      <span className="flex items-center gap-1.5 rounded-full bg-slate-800/60 border border-slate-700/60 px-3 py-1 text-xs text-slate-400">
        <span className="h-2 w-2 rounded-full bg-slate-500 animate-pulse" />
        Connecting…
      </span>
    );
  }
  return (
    <span
      className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${
        ok
          ? "bg-emerald-950/50 border-emerald-700/50 text-emerald-400"
          : "bg-red-950/50 border-red-700/50 text-red-400"
      }`}
    >
      {ok ? (
        <CheckCircle2 className="h-3 w-3" />
      ) : (
        <XCircle className="h-3 w-3" />
      )}
      {ok ? labelOk : labelFail}
    </span>
  );
}
