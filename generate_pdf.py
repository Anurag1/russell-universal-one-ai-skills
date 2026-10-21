#!/usr/bin/env python3
"""Generate a clean social-media / shareable PDF for Russell Universal One AI Skills."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

BG_DARK = HexColor("#0b0f19")
ACCENT = HexColor("#34d399")
ACCENT2 = HexColor("#38bdf8")
TEXT = HexColor("#f1f5f9")
MUTED = HexColor("#94a3b8")
BORDER = HexColor("#1e293b")

OUTPUT = "Russell_Universal_One_AI_Skills.pdf"

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        rightMargin=18*mm,
        leftMargin=18*mm,
        topMargin=16*mm,
        bottomMargin=16*mm,
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="HeroTitle", fontName="Helvetica-Bold", fontSize=20,
        textColor=ACCENT, alignment=TA_CENTER, spaceAfter=6, leading=24,
    ))
    styles.add(ParagraphStyle(
        name="HeroSub", fontName="Helvetica", fontSize=10,
        textColor=MUTED, alignment=TA_CENTER, spaceAfter=10, leading=13,
    ))
    styles.add(ParagraphStyle(
        name="Section", fontName="Helvetica-Bold", fontSize=12,
        textColor=ACCENT, spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Body", fontName="Helvetica", fontSize=9,
        textColor=TEXT, leading=12, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock", fontName="Courier", fontSize=7.5,
        textColor=HexColor("#a5f3fc"), leading=10, leftIndent=4, rightIndent=4,
    ))
    styles.add(ParagraphStyle(
        name="CardTitle", fontName="Helvetica-Bold", fontSize=9.5,
        textColor=TEXT, spaceBefore=2, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="Highlight", fontName="Helvetica", fontSize=9,
        textColor=TEXT, leading=12, leftIndent=6, rightIndent=6,
    ))
    styles.add(ParagraphStyle(
        name="Footer", fontName="Helvetica", fontSize=8,
        textColor=MUTED, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="Link", fontName="Helvetica", fontSize=8.5,
        textColor=ACCENT2, alignment=TA_CENTER, spaceBefore=2, spaceAfter=2,
    ))

    story = []

    story.append(Paragraph("Russell Universal One → Modern AI Skills", styles["HeroTitle"]))
    story.append(Paragraph(
        "Six novel AI primitives inspired by Walter Russell’s <i>The Universal One</i> (1926).<br/>"
        "Unified light field · Polarity balance · Octave dynamics · Dimensionless core",
        styles["HeroSub"]
    ))
    story.append(Paragraph(
        "<b>Open Source</b>  ·  <b>Live Demo</b>  ·  <b>PyTorch FNO</b>  ·  <b>Residual → 0</b>",
        styles["HeroSub"]
    ))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=8))

    story.append(Paragraph(
        "<b>Key result:</b> Polarity residual driven to <font color='#34d399'><b>0.000000</b></font> while learning. "
        "A continuous “light” field projects into language, vision and physics. "
        "A tiny dimensionless core controls rich appearance.",
        styles["Highlight"]
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "Live Demo:  https://anurag1.github.io/russell-universal-one-ai-skills/<br/>"
        "GitHub:     https://github.com/Anurag1/russell-universal-one-ai-skills",
        styles["Link"]
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))

    story.append(Paragraph("Skills 3–6 — Live Output", styles["Section"]))

    story.append(Paragraph("3. Polarized Generation + Balance  <font color='#34d399'>[Residual → 0]</font>", styles["CardTitle"]))
    story.append(Preformatted("""Final positive stream : [ 0.482 -0.127  0.455  0.632 -0.383 -0.34   0.57   0.5  ]
Final negative stream : [-0.482  0.127 -0.455 -0.632  0.383  0.34  -0.57  -0.5  ]
Balance residuals     : [0.04859, 0.03888, 0.03986, 0.02456, 0.04013]
Final residual energy : 0.000000 (near zero = balanced)

→ Intrinsic anti-hallucination via forced polarity cancellation.""", styles["CodeBlock"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("4. Dimensionless Core → Dimensional Appearance  <font color='#34d399'>[Core]</font>", styles["CardTitle"]))
    story.append(Preformatted("""Dimensionless core     : [-0.53   0.513  0.097  0.969]
Projected 3-D + mass   : [-0.656 -0.029 -0.517  1.221]

→ Any high-dimensional world can be collapsed back to the same core.""", styles["CodeBlock"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("5. Simple Mind Cause → Complex Appearance  <font color='#34d399'>[3 Params]</font>", styles["CardTitle"]))
    story.append(Preformatted("""Cause A (pos, oct1) → effect mean/std : 0.000 / 0.764
Cause B (neg, oct2) → effect mean/std : -0.000 / 0.680
Effect A sample     : [ 0.     1.409 -0.074 -1.243  0.315  0.618]
Effect B sample     : [ 0.    -0.11   0.022 -0.384  0.167 -0.747]

→ Tiny causal parameter set fully determines rich, controllable appearance.""", styles["CodeBlock"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("6. Integrated Mini-Agent — One Thinking Step  <font color='#34d399'>[Full Cycle]</font>", styles["CardTitle"]))
    story.append(Preformatted("""Input vector (first 6) : [-0.161  0.404  1.886  0.175  0.258 -0.074]
Octave energies        : [0.9403, 0.8713, 0.976]
Balance residual       : 0.167181
Dimensionless core     : [0.0293, -0.0293, 0.1182]
New appearance sample  : [-0.001, 0.003, 0.013, 0.001, 0.002, -0.001]

→ Full cycle: light field → octave analysis → polarity balance
  → dimensionless core → new dimensional appearance.""", styles["CodeBlock"]))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=6))

    story.append(Paragraph("Also Included", styles["Section"]))
    story.append(Paragraph(
        "• <b>Unified One-Substance Wave Field</b> — single continuous field → language + vision + physics<br/>"
        "• <b>Octave / Periodic Cycle Dynamics</b> — multi-scale hierarchical processing<br/>"
        "• <b>Russell-Enhanced Fourier Neural Operator</b> — polarity residual + dimensionless core added to classic FNO (live-tested on CPU)",
        styles["Body"]
    ))

    story.append(Paragraph("Highest-Impact Targets", styles["Section"]))
    story.append(Paragraph(
        "• World Models (Genie, Cosmos…)<br/>"
        "• Neural Operators / Physics-informed ML<br/>"
        "• Frontier multi-modal foundation models<br/>"
        "• Robotics foundation models (VLA → World-Action)",
        styles["Body"]
    ))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=8))

    story.append(Paragraph(
        "Inspired by Walter Russell’s <i>The Universal One</i> (1926)<br/>"
        "Open source · MIT License · github.com/Anurag1/russell-universal-one-ai-skills",
        styles["Footer"]
    ))

    def add_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BG_DARK)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    from reportlab.lib.pagesizes import A4
    doc.build(story, onFirstPage=add_bg, onLaterPages=add_bg)
    print(f"PDF generated: {OUTPUT}")

if __name__ == "__main__":
    build_pdf()
