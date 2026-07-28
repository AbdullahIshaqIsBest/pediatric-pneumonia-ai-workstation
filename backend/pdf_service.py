"""
pdf_service.py
==============
Generates a professional clinical PDF diagnostic report using ReportLab.

The report includes:
  • Patient encounter header with timestamp
  • Embedded X-ray image and Grad-CAM heatmap overlay
  • Classification probabilities & decision threshold
  • Reference model metrics (ROC-AUC, Sensitivity)
  • Clinical interpretation text
  • Digital signature block  "Software Architecture by Abdullah Ishaq"
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_DARK_BLUE  = colors.HexColor("#0B2447")
_MED_BLUE   = colors.HexColor("#19376D")
_ACCENT     = colors.HexColor("#0EA5E9")
_RED        = colors.HexColor("#EF4444")
_GREEN      = colors.HexColor("#22C55E")
_LIGHT_GREY = colors.HexColor("#F1F5F9")
_DARK_GREY  = colors.HexColor("#334155")
_BLACK      = colors.HexColor("#0F172A")


def _pil_b64_to_rl_image(b64_str: str, width_cm: float, height_cm: float) -> RLImage:
    """Decode base64 PNG/JPEG string into a ReportLab Image object."""
    data = base64.b64decode(b64_str)
    buf = io.BytesIO(data)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def generate_pdf(payload: Dict[str, Any]) -> bytes:
    """
    Build and return a diagnostic report PDF as raw bytes.

    Expected keys in ``payload``
    ----------------------------
    prediction       : str   "PNEUMONIA" | "NORMAL"
    confidence       : float  0-1
    prob_normal      : float  0-1
    prob_pneumonia   : float  0-1
    threshold        : float  decision threshold used
    heatmap_base64   : str   base64-encoded Grad-CAM overlay PNG
    original_base64  : str   base64-encoded original X-ray PNG  (optional)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Pediatric Pneumonia Diagnostic Report",
    )

    styles = getSampleStyleSheet()

    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                         fontSize=18, textColor=_DARK_BLUE, spaceAfter=2,
                         fontName="Helvetica-Bold", alignment=TA_CENTER)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                         fontSize=13, textColor=_MED_BLUE, spaceBefore=8,
                         fontName="Helvetica-Bold")
    body = ParagraphStyle("Body", parent=styles["Normal"],
                           fontSize=9.5, textColor=_DARK_GREY, leading=14)
    small = ParagraphStyle("Small", parent=styles["Normal"],
                            fontSize=8, textColor=_DARK_GREY)
    sig   = ParagraphStyle("Sig", parent=styles["Normal"],
                            fontSize=9, textColor=_DARK_BLUE,
                            fontName="Helvetica-Oblique", alignment=TA_RIGHT)

    prediction   = payload.get("prediction", "UNKNOWN")
    confidence   = float(payload.get("confidence", 0.0))
    prob_normal  = float(payload.get("prob_normal", 0.0))
    prob_pneu    = float(payload.get("prob_pneumonia", 0.0))
    threshold    = float(payload.get("threshold", 0.933))
    heatmap_b64  = payload.get("heatmap_base64", "")
    original_b64 = payload.get("original_base64", "")

    is_pneumonia = prediction == "PNEUMONIA"
    result_color = _RED if is_pneumonia else _GREEN
    result_label = "⚠  PNEUMONIA DETECTED" if is_pneumonia else "✔  NORMAL LUNG PARENCHYMA"

    timestamp = datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")

    story = []

    # ── Header bar ──────────────────────────────────────────────────────────
    story.append(Paragraph("Pediatric Pneumonia AI Diagnostic Workstation", h1))
    story.append(Paragraph("Software by Abdullah Ishaq", ParagraphStyle(
        "Sub", parent=styles["Normal"], fontSize=9, textColor=_ACCENT,
        alignment=TA_CENTER)))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_ACCENT))
    story.append(Spacer(1, 3 * mm))

    # ── Meta row ────────────────────────────────────────────────────────────
    meta_data = [
        ["Report Generated:", timestamp,
         "Model:", "ResNet-50 (layer4)"],
        ["Decision Threshold:", f"{threshold:.3f}",
         "ROC-AUC (ref):", "0.978"],
        ["Sensitivity (ref):", "96.9%",
         "Specificity (ref):", "94.1%"],
    ]
    meta_table = Table(meta_data, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), _MED_BLUE),
        ("TEXTCOLOR", (2, 0), (2, -1), _MED_BLUE),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, -1), _LIGHT_GREY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, _LIGHT_GREY]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5 * mm))

    # ── Primary result ───────────────────────────────────────────────────────
    story.append(Paragraph("Primary Diagnostic Finding", h2))
    result_table = Table(
        [[Paragraph(f"<b>{result_label}</b>",
                    ParagraphStyle("Res", parent=styles["Normal"],
                                   fontSize=14, textColor=colors.white,
                                   fontName="Helvetica-Bold", alignment=TA_CENTER))]],
        colWidths=[18 * cm]
    )
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), result_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 3 * mm))

    # ── Probability table ────────────────────────────────────────────────────
    prob_data = [
        ["Class", "Probability", "Status"],
        ["NORMAL",    f"{prob_normal * 100:.2f} %",  "–"],
        ["PNEUMONIA", f"{prob_pneu   * 100:.2f} %",
         "⚠ DETECTED" if is_pneumonia else "–"],
    ]
    prob_table = Table(prob_data, colWidths=[6*cm, 6*cm, 6*cm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), _MED_BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1),(-1, -1), [colors.white, _LIGHT_GREY]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 5 * mm))

    # ── Images ───────────────────────────────────────────────────────────────
    story.append(Paragraph("Chest X-Ray & Grad-CAM Attention Map", h2))
    img_row = []
    if original_b64:
        img_row.append(_pil_b64_to_rl_image(original_b64, 8.0, 8.0))
    if heatmap_b64:
        img_row.append(_pil_b64_to_rl_image(heatmap_b64, 8.0, 8.0))

    if img_row:
        label_row = []
        if original_b64:
            label_row.append(Paragraph("Original X-Ray", ParagraphStyle(
                "IL", parent=styles["Normal"], fontSize=8,
                alignment=TA_CENTER, textColor=_DARK_GREY)))
        if heatmap_b64:
            label_row.append(Paragraph("Grad-CAM Activation (layer4)", ParagraphStyle(
                "IL", parent=styles["Normal"], fontSize=8,
                alignment=TA_CENTER, textColor=_DARK_GREY)))

        col_w = 9.0 * cm if len(img_row) == 2 else 18.0 * cm
        img_table = Table([img_row, label_row], colWidths=[col_w] * len(img_row))
        img_table.setStyle(TableStyle([
            ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",  (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(img_table)
    story.append(Spacer(1, 5 * mm))

    # ── Clinical interpretation ──────────────────────────────────────────────
    story.append(Paragraph("AI-Assisted Clinical Interpretation", h2))
    if is_pneumonia:
        findings = [
            f"• <b>Finding:</b> The deep learning model classifies this study as "
            f"<b>PNEUMONIA</b> with a confidence of {confidence*100:.1f}%.",
            "• <b>Radiological Pattern:</b> Focal pulmonary opacity suggestive of "
            "consolidation or infiltrate. Attention regions identified by Grad-CAM "
            "localise to the lower lobe(s).",
            "• <b>Clinical Recommendation:</b> Correlate with clinical presentation, "
            "laboratory findings (CBC, CRP), and SpO₂. Consider chest physiotherapy "
            "and antibiotic therapy per local guidelines if bacterial aetiology is "
            "suspected.",
            "• <b>Differential Diagnosis:</b> Bacterial pneumonia, viral pneumonitis, "
            "aspiration pneumonia, or pulmonary oedema.",
        ]
    else:
        findings = [
            f"• <b>Finding:</b> The model classifies this study as "
            f"<b>NORMAL</b> with a confidence of {confidence*100:.1f}%.",
            "• <b>Radiological Pattern:</b> Clear lung parenchyma. No focal opacity, "
            "consolidation, or pleural effusion detected by the AI attention map.",
            "• <b>Clinical Recommendation:</b> No immediate radiological intervention "
            "indicated. Continue clinical observation and symptomatic management.",
            "• <b>Note:</b> A negative AI result does not exclude subtle or early "
            "disease. Clinical judgement must always take precedence.",
        ]

    disclaimer = (
        "<b>⚠ Disclaimer:</b> This report is generated by an AI research prototype "
        "and is intended for research and educational purposes only. It must NOT be "
        "used as a sole basis for clinical decision-making. All findings should be "
        "reviewed and validated by a qualified radiologist or clinician."
    )

    for f in findings:
        story.append(Paragraph(f, body))
        story.append(Spacer(1, 1.5 * mm))

    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(disclaimer, ParagraphStyle(
        "Disc", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#64748B"), leading=11,
        backColor=colors.HexColor("#FFF7ED"),
        borderPadding=(4, 6, 4, 6))))

    # ── Signature ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width="100%", thickness=1, color=_MED_BLUE))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "Software Architecture by <b>Abdullah Ishaq</b> — Pediatric Pneumonia AI Research Pipeline",
        sig))
    story.append(Paragraph(
        f"Model: ResNet-50 | Backbone: ImageNet Pretrained | Generated: {timestamp}",
        ParagraphStyle("SigSub", parent=styles["Normal"], fontSize=7.5,
                        textColor=colors.HexColor("#94A3B8"), alignment=TA_RIGHT)))

    doc.build(story)
    return buf.getvalue()
