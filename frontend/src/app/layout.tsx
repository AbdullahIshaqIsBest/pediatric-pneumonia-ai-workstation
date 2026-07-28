import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Pediatric Pneumonia AI Diagnostic Workstation",
  description:
    "Production-grade AI system for classifying pediatric chest X-rays using ResNet-50 with Grad-CAM attention visualization. Software by Abdullah Ishaq.",
  keywords: [
    "pneumonia detection",
    "chest X-ray AI",
    "medical imaging",
    "deep learning",
    "ResNet-50",
    "Grad-CAM",
  ],
  authors: [{ name: "Abdullah Ishaq" }],
  openGraph: {
    title: "Pediatric Pneumonia AI Diagnostic Workstation",
    description:
      "AI-powered chest X-ray classification with Grad-CAM attention visualization.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
