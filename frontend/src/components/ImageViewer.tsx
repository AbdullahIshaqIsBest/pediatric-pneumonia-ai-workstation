"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  UploadCloud,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  SunMedium,
  Contrast,
  FlipHorizontal2,
  Layers,
} from "lucide-react";
import { ImageControls } from "@/types";

interface Props {
  onFileSelect: (file: File, base64: string) => void;
  heatmapBase64: string | null;
  controls: ImageControls;
  onControlChange: (key: keyof ImageControls, value: number | boolean) => void;
}

const DEFAULT_CONTROLS: ImageControls = {
  brightness: 100,
  contrast: 100,
  invert: false,
  heatmapOpacity: 0,
  zoom: 1,
};

export default function ImageViewer({
  onFileSelect,
  heatmapBase64,
  controls,
  onControlChange,
}: Props) {
  const [dragging, setDragging] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const hmRef = useRef<HTMLImageElement | null>(null);
  const panStart = useRef<{ x: number; y: number } | null>(null);
  const panOffset = useRef({ x: 0, y: 0 });
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load heatmap image object when base64 changes
  useEffect(() => {
    if (!heatmapBase64) {
      setHeatmapUrl(null);
      hmRef.current = null;
      return;
    }
    const url = `data:image/png;base64,${heatmapBase64}`;
    setHeatmapUrl(url);
    const hm = new window.Image();
    hm.src = url;
    hm.onload = () => {
      hmRef.current = hm;
      redraw();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [heatmapBase64]);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgRef.current) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cw = canvas.width;
    const ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);

    // Apply CSS filter via canvas filter API
    const filterStr = [
      `brightness(${controls.brightness}%)`,
      `contrast(${controls.contrast}%)`,
      controls.invert ? "invert(1)" : "",
    ]
      .filter(Boolean)
      .join(" ");

    ctx.save();
    ctx.translate(panOffset.current.x + cw / 2, panOffset.current.y + ch / 2);
    ctx.scale(controls.zoom, controls.zoom);

    // Draw base image
    const img = imgRef.current;
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
    const dw = img.naturalWidth * scale;
    const dh = img.naturalHeight * scale;
    ctx.filter = filterStr || "none";
    ctx.drawImage(img, -dw / 2, -dh / 2, dw, dh);

    // Draw heatmap overlay
    if (hmRef.current && controls.heatmapOpacity > 0) {
      ctx.filter = "none";
      ctx.globalAlpha = controls.heatmapOpacity / 100;
      ctx.drawImage(hmRef.current, -dw / 2, -dh / 2, dw, dh);
      ctx.globalAlpha = 1;
    }

    ctx.restore();
  }, [controls]);

  // Redraw whenever controls change
  useEffect(() => {
    redraw();
  }, [controls, redraw]);

  const loadImage = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string;
        setImageUrl(dataUrl);

        const img = new window.Image();
        img.src = dataUrl;
        img.onload = () => {
          imgRef.current = img;
          panOffset.current = { x: 0, y: 0 };
          redraw();
        };
        onFileSelect(file, dataUrl.split(",")[1] ?? "");
      };
      reader.readAsDataURL(file);
    },
    [onFileSelect, redraw]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) loadImage(file);
    },
    [loadImage]
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadImage(file);
  };

  // Pan
  const onMouseDown = (e: React.MouseEvent) => {
    panStart.current = { x: e.clientX - panOffset.current.x, y: e.clientY - panOffset.current.y };
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!panStart.current) return;
    panOffset.current = {
      x: e.clientX - panStart.current.x,
      y: e.clientY - panStart.current.y,
    };
    redraw();
  };
  const onMouseUp = () => { panStart.current = null; };

  // Scroll zoom
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.1 : -0.1;
    const newZoom = Math.max(0.5, Math.min(5, controls.zoom + delta));
    onControlChange("zoom", newZoom);
  };

  const resetView = () => {
    panOffset.current = { x: 0, y: 0 };
    onControlChange("zoom", 1);
    onControlChange("brightness", 100);
    onControlChange("contrast", 100);
    onControlChange("invert", false);
    onControlChange("heatmapOpacity", 0);
  };

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Canvas area */}
      <div
        ref={containerRef}
        className={`relative upload-zone flex-1 min-h-[400px] rounded-xl border-2 border-dashed border-slate-700/60 bg-slate-900/40 overflow-hidden flex items-center justify-center ${
          dragging ? "dragging" : ""
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        {!imageUrl ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center gap-3 text-center px-6 select-none"
          >
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-800/80 border border-slate-700/60">
              <UploadCloud className="h-8 w-8 text-slate-400" />
            </div>
            <div>
              <p className="font-semibold text-slate-300 text-sm">
                Drag & drop a chest X-ray here
              </p>
              <p className="text-xs text-slate-500 mt-1">
                PNG, JPG, JPEG supported
              </p>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="mt-1 rounded-lg bg-cyan-600/80 hover:bg-cyan-600 px-4 py-2 text-xs font-semibold text-white transition-colors"
            >
              Browse Files
            </button>
          </motion.div>
        ) : (
          <canvas
            ref={canvasRef}
            width={600}
            height={500}
            className="w-full h-full grab-cursor"
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
            onWheel={onWheel}
          />
        )}

        {/* Drag overlay */}
        <AnimatePresence>
          {dragging && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-cyan-500/10 flex items-center justify-center rounded-xl border-2 border-cyan-500/60"
            >
              <p className="text-cyan-400 font-semibold text-lg">Drop to Analyze</p>
            </motion.div>
          )}
        </AnimatePresence>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg"
          className="hidden"
          onChange={handleFileInput}
        />
      </div>

      {/* Controls bar */}
      {imageUrl && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-4 flex flex-col gap-3"
        >
          {/* Zoom & quick buttons */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[11px] text-slate-400 uppercase tracking-widest font-semibold w-full">
              Image Controls
            </span>

            <button
              onClick={() => onControlChange("zoom", Math.min(5, controls.zoom + 0.25))}
              title="Zoom In"
              className="rounded-lg bg-slate-800/60 border border-slate-700/40 p-2 text-slate-300 hover:text-white hover:border-cyan-500/40 transition-colors"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
            <button
              onClick={() => onControlChange("zoom", Math.max(0.5, controls.zoom - 0.25))}
              title="Zoom Out"
              className="rounded-lg bg-slate-800/60 border border-slate-700/40 p-2 text-slate-300 hover:text-white hover:border-cyan-500/40 transition-colors"
            >
              <ZoomOut className="h-4 w-4" />
            </button>
            <button
              onClick={() => onControlChange("invert", !controls.invert)}
              title="Invert Contrast"
              className={`rounded-lg border p-2 transition-colors ${
                controls.invert
                  ? "bg-cyan-600/30 border-cyan-500/60 text-cyan-300"
                  : "bg-slate-800/60 border-slate-700/40 text-slate-300 hover:text-white"
              }`}
            >
              <FlipHorizontal2 className="h-4 w-4" />
            </button>
            <button
              onClick={resetView}
              title="Reset all"
              className="rounded-lg bg-slate-800/60 border border-slate-700/40 p-2 text-slate-300 hover:text-white hover:border-red-500/40 transition-colors"
            >
              <RotateCcw className="h-4 w-4" />
            </button>

            <span className="ml-auto text-xs text-slate-500 font-mono">
              {(controls.zoom * 100).toFixed(0)}%
            </span>
          </div>

          {/* Sliders */}
          <div className="grid gap-2.5">
            <Slider
              label="Brightness"
              icon={<SunMedium className="h-3.5 w-3.5 text-amber-400" />}
              value={controls.brightness}
              min={20}
              max={200}
              unit="%"
              onChange={(v) => onControlChange("brightness", v)}
            />
            <Slider
              label="Contrast"
              icon={<Contrast className="h-3.5 w-3.5 text-indigo-400" />}
              value={controls.contrast}
              min={20}
              max={300}
              unit="%"
              onChange={(v) => onControlChange("contrast", v)}
            />
            <Slider
              label="Grad-CAM Opacity"
              icon={<Layers className="h-3.5 w-3.5 text-cyan-400" />}
              value={controls.heatmapOpacity}
              min={0}
              max={100}
              unit="%"
              disabled={!heatmapBase64}
              onChange={(v) => onControlChange("heatmapOpacity", v)}
            />
          </div>
        </motion.div>
      )}
    </div>
  );
}

function Slider({
  label,
  icon,
  value,
  min,
  max,
  unit,
  disabled,
  onChange,
}: {
  label: string;
  icon: React.ReactNode;
  value: number;
  min: number;
  max: number;
  unit: string;
  disabled?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <div className={`flex items-center gap-3 ${disabled ? "opacity-40 pointer-events-none" : ""}`}>
      <span className="shrink-0">{icon}</span>
      <span className="text-xs text-slate-400 w-28 shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        className="flex-1 h-1"
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="text-xs font-mono text-slate-300 w-12 text-right">
        {value}{unit}
      </span>
    </div>
  );
}
