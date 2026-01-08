from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import date

def generate_pdf(path, city, district, pack):
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h-40, "Barzel Analytics – Screening Memo")

    c.setFont("Helvetica", 10)
    c.drawString(40, h-65, f"{city} / {district} — {date.today().isoformat()}")
    c.drawString(40, h-80, "Screening memo (not underwriting).")

    y = h-120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Decision")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(40, y, "Decision derived from structured screening metrics.")

    y -= 28
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Rationale")
    y -= 18
    d = pack["descriptors"]
    p = pack["proxies"]
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Pricing level (median AED/sqm): {int(d['price_p50']) if d['price_p50'] else 'n/a'}")
    y -= 14
    c.drawString(40, y, f"Dispersion (IQR): {int(d['price_iqr']) if d['price_iqr'] else 'n/a'}")
    y -= 14
    c.drawString(40, y, f"Liquidity proxy (DOM median): {int(d['dom_p50']) if d['dom_p50'] else 'n/a'}")

    y -= 28
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Limitations")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(40, y, "Listings-based data, proxies only, no transaction confirmation.")

    c.showPage()
    c.save()
