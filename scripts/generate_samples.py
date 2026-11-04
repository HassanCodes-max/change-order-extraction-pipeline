"""Generate synthetic change-order documents into ``sample_documents/``.

All companies, people and projects are fictional. Each sample targets a
specific difficulty:

=============================  ==============================================
co_014_standard.pdf            clean two-column form, consistent arithmetic
co_007_scanned.pdf             image-only (no text layer) scan -> OCR path
co_022_missing_fields.txt      most optional fields absent
co_031_conflicting.pdf         conflicting CO numbers / schedule, bad arithmetic
co_045_messy_format.txt        lowercase labels, dot leaders, (credit), invalid date
co_052_narrative_email.txt     no labels at all: AIA-style prose in an email
=============================  ==============================================

Run:  python -m scripts.generate_samples
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import pymupdf  # PyMuPDF
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "sample_documents"

# A tab separates the label column from the value column in the layouts below.
CO_014_STANDARD = """\
SUMMIT RIDGE BUILDERS, LLC
1450 Industrial Parkway, Denver, CO 80216

CHANGE ORDER
Change Order No.:\tCO-014
Date:\tMarch 3, 2025
Project:\tABC Commercial Building
Project No.:\t2024-118
Original Contract No.:\tGC-2024-0457
Contractor:\tSummit Ridge Builders, LLC
Owner:\tABC Holdings Inc.

Description of Change:
Furnish and install twelve (12) additional LED high-bay fixtures in the
warehouse area, including conduit, wiring and circuiting to panel LP-2.

Reason for Change: Owner request to increase lighting levels in the warehouse.

Original Contract Amount:\t$1,250,000.00
Additional Cost:\t$12,500.00
Deducted Amount:\t$0.00
Revised Contract Amount:\t$1,262,500.00

Schedule Impact: Project completion extended by five (5) calendar days.
Requested Completion Date:\t08/15/2025
Status:\tApproved

Prepared by: Jane Doe, Project Manager
Approved by: Robert Smith, Owner's Representative
"""

CO_007_SCANNED = """\
RIVERSIDE MEDICAL OFFICE BUILDING
CHANGE ORDER

Change Order No:\tCO-007
Date:\tApril 22, 2025
Project:\tRiverside Medical Office Building
Project No:\tRMB-2291
Contract No:\tC-88120
Contractor:\tKeystone Mechanical Inc.
Owner:\tRiverside Health Partners

Description of Change:
Replace specified rooftop unit RTU-3 with a 25-ton high-efficiency
unit and add a new curb adapter.

Reason for Change: Specified unit discontinued by manufacturer.

Original Contract Amount:\t$842,300.00
Additional Cost:\t$18,750.00
Credit:\t$2,100.00
Revised Contract Amount:\t$858,950.00

Schedule Impact:\t10 calendar days
Requested Completion Date:\tSeptember 30, 2025
Status:\tPending
Prepared by:\tLuis Ortega
"""

CO_022_MISSING = """\
Change Order Request

CO #: 22
Project: Westgate Elementary School Renovation
Date: 2025-06-10
Contractor: Brightline Construction Co.

Scope of Work:
Remove and dispose of additional asbestos-containing floor tile discovered
beneath existing carpet in classrooms 104-108 (approx. 1,200 sq ft).

Additional Cost: $9,850.00

Status: Submitted for owner review
"""

CO_031_CONFLICTING = """\
CHANGE ORDER No. CO-031
Northgate Logistics Center - Building B

Change Order Number:\tCO-013
Date:\t07/18/2025
Project:\tNorthgate Logistics Center
Project No.:\tNLC-B-0098
Contract No.:\tNLC-GC-2024-12
Contractor:\tIronwood General Contractors
Owner:\tNorthgate Industrial REIT

Description of Change:
Upgrade dock levelers at doors 14-22 from 30,000 lb to 45,000 lb capacity
mechanical levelers, including pit modifications.

Reason for Change: Tenant equipment requirements.

Original Contract Amount:\t$1,000,000.00
Additional Cost:\t$45,000.00
Revised Contract Amount:\t$1,054,000.00

Schedule Impact:\t10 days
The Contract Time will be extended by fourteen (14) calendar days.

Status:\tApproved
Approved by:\t________________________
"""

CO_045_MESSY = """\
*** CHANGE ORDER ***
proj:  harbor view condos - phase 2        job # HV-PH2-17
co no. 45        dated 3rd of March 2025
contract ref - HVC/2023/114
GENERAL CONTRACTOR -- Atlas Building Group
client -- Harbor View Development LLC

description of changes:
  - relocate 6 floor drains in level P1 garage per revised civil drawings C-201 rev 4
  - add trench drain at ramp entry (40 LF)

justification: civil drawing revision issued after bid

orig. contract sum......USD 3,480,000
add amount.............. USD 27,400.50
deduct.................. (3,200.00)
new contract total...... USD 3,504,200.50

time extension: two weeks
completion date: 02/30/2026
STATUS: approved as noted
submitted by: m. chen / atlas
approved by: D. Patel
"""

CO_052_NARRATIVE = """\
Subject: Change Order 52 - Lobby Millwork

Hi Karen,

Following our walkthrough on May 14, 2025, Pinnacle Interiors proposes to replace the
specified oak veneer at the main lobby reception desk with walnut veneer, per the
owner's revised finish selection.

The Contract Sum will be increased by this Change Order in the amount of $6,420.00.
The original Contract Sum was $512,000.00. The new Contract Sum including this Change
Order will be $518,420.00. The Contract Time will be increased by three (3) days.

Please countersign below to approve.

Regards,
Tom Becker
Pinnacle Interiors
Project: Lakeshore Tower Lobby Refresh
"""

TEXT_SAMPLES = {
    "co_022_missing_fields.txt": CO_022_MISSING,
    "co_045_messy_format.txt": CO_045_MESSY,
    "co_052_narrative_email.txt": CO_052_NARRATIVE,
}

PAGE_W, PAGE_H = 612, 792  # US Letter, points
LEFT, VALUE_X, TOP, LINE_H, FONT_SIZE = 54, 230, 60, 15, 10


def make_text_pdf(text: str, path: Path | None = None) -> bytes:
    """Render ``text`` as a PDF with a real text layer (tab = value column)."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = TOP
    for line in text.splitlines():
        if y > PAGE_H - 50:
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = TOP
        if "\t" in line:
            label, value = line.split("\t", 1)
            page.insert_text((LEFT, y), label, fontsize=FONT_SIZE, fontname="helv")
            page.insert_text((VALUE_X, y), value, fontsize=FONT_SIZE, fontname="helv")
        elif line.strip():
            bold = line.isupper()
            page.insert_text((LEFT, y), line, fontsize=FONT_SIZE + (2 if bold else 0),
                             fontname="hebo" if bold else "helv")
        y += LINE_H
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    if path:
        path.write_bytes(data)
    return data


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/Library/Fonts/Arial.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_scan(text: str, dpi: int = 200, noise: float = 0.02, rotation: float = 0.6,
                blur: float = 0.6, seed: int = 7) -> Image.Image:
    """Render ``text`` to an image that imitates an office scanner."""
    scale = dpi / 72
    width, height = int(PAGE_W * scale), int(PAGE_H * scale)
    img = Image.new("L", (width, height), 250)
    draw = ImageDraw.Draw(img)
    font = _load_font(int(FONT_SIZE * scale * 1.05))
    y = TOP * scale
    for line in text.splitlines():
        if "\t" in line:
            label, value = line.split("\t", 1)
            draw.text((LEFT * scale, y), label, fill=25, font=font)
            draw.text((VALUE_X * scale, y), value, fill=25, font=font)
        elif line.strip():
            draw.text((LEFT * scale, y), line, fill=25, font=font)
        y += LINE_H * scale
    img = img.rotate(rotation, resample=Image.BICUBIC, fillcolor=250)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    rng = random.Random(seed)
    px = img.load()
    for _ in range(int(width * height * noise)):
        x, yy = rng.randrange(width), rng.randrange(height)
        px[x, yy] = rng.choice((30, 90, 200))
    return img


def make_scanned_pdf(text: str, path: Path | None = None, **scan_kwargs) -> bytes:
    """Create an image-only PDF (no text layer), as produced by a scanner."""
    img = render_scan(text, **scan_kwargs)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(page.rect, stream=buf.getvalue())
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    if path:
        path.write_bytes(data)
    return data


def generate(out_dir: Path = SAMPLES_DIR) -> list[Path]:
    """Write all samples to ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in (("co_014_standard.pdf", CO_014_STANDARD),
                          ("co_031_conflicting.pdf", CO_031_CONFLICTING)):
        make_text_pdf(content, out_dir / name)
        written.append(out_dir / name)
    make_scanned_pdf(CO_007_SCANNED, out_dir / "co_007_scanned.pdf")
    written.append(out_dir / "co_007_scanned.pdf")
    for name, content in TEXT_SAMPLES.items():
        (out_dir / name).write_text(content.replace("\t", "    "), encoding="utf-8")
        written.append(out_dir / name)
    return written


if __name__ == "__main__":
    for p in generate():
        print(f"wrote {p.relative_to(ROOT)}")
