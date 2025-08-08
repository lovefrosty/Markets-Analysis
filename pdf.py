
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from io import BytesIO
import textwrap
from typing import Dict, Any

def _draw_wrapped(c, text, x, y, max_width, leading=14, font="Helvetica", size=10):
    c.setFont(font, size)
    wrapped = []
    for line in text.split("\n"):
        wrapped += textwrap.wrap(line, width=int(max_width/6)) or [""]
    for w in wrapped:
        c.drawString(x, y, w)
        y -= leading
    return y

def render_pdf(context: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 0.75*inch
    y = height - margin
    c.setTitle(f"{context.get('ticker','TICKER')} Research Note")

    def heading(txt, size=14):
        nonlocal y
        c.setFont("Helvetica-Bold", size)
        c.drawString(margin, y, txt)
        y -= 18

    def kv(label, value):
        nonlocal y
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"{label}: {value}")
        y -= 12

    heading(f"{context.get('ticker','TICKER')} — Automated Equity Research Note")
    kv("Generated", context.get("as_of",""))
    y -= 6

    heading("Snapshot")
    snap = context["snapshot"]; risk = context["risk"]; opts = context["options"]
    kv("Last price", f"${snap['last_price']:.2f}")
    kv("Beta", f"{risk.get('beta','N/A')}")
    kv("Risk-free (10Y)", f"{risk.get('rf',0)*100:.2f}%")
    kv("Expected return (CAPM)", f"{risk.get('expected_return',0)*100:.2f}%")
    kv("Implied 1w move (ATM)", f"±{opts.get('expected_move',0)*100:.2f}%")
    y -= 6

    heading("Valuation")
    dcf = context["valuation"]["dcf"]
    kv("Fair value / share", f"${dcf.get('fair_value_per_share','N/A'):.2f}" if dcf.get('fair_value_per_share') else "N/A")
    kv("Upside", f"{dcf.get('upside_pct',0)*100:.1f}%")
    y -= 6

    heading("Reverse DCF")
    rdcf = context["valuation"]["reverse_dcf"]
    kv("Implied Rev CAGR(1–5y)", f"{rdcf.get('implied_revenue_cagr',0)*100:.2f}%")
    kv("Implied Op Margin", f"{rdcf.get('implied_op_margin',0)*100:.1f}%")
    y -= 6

    heading("Comps Snapshot")
    comps = context.get("comps_summary", {})
    for k,v in comps.items():
        kv(k, v)
    y -= 6

    heading("Risk-Neutral Density (nearest expiry)")
    rnd = context.get("rnd", {})
    for k,v in rnd.items():
        kv(k, v)
    y -= 6

    heading("ML Targets")
    ml = context.get("ml", {})
    for k,v in ml.items():
        kv(k, v)

    c.showPage()
    c.save()
    return buffer.getvalue()
