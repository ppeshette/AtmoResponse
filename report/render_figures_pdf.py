"""Render FIGURES.md and the figures/*.png into FIGURES.pdf, one figure per block.

The committed FIGURES.pdf is produced by this script. Regenerate it after editing
FIGURES.md or replacing a figure PNG:

    pip install reportlab
    python report/render_figures_pdf.py

Uses reportlab's built-in Times family, so it needs no system fonts and runs on
any platform.
"""

from __future__ import annotations

import html
import io
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "FIGURES.md"
OUTPUT = HERE / "FIGURES.pdf"
MARGIN = 0.75 * inch
CONTENT_WIDTH = LETTER[0] - 2 * MARGIN

FIGURE_RE = re.compile(r"`figures/([A-Za-z0-9_]+\.png)`\.\s*")


def inline_markdown(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return re.sub(r"(?<!\*)\*([^*]+)\*", r"<i>\1</i>", escaped)


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Title"], fontName="Times-Bold", fontSize=18,
            leading=22, spaceAfter=10,
        ),
        "lead": ParagraphStyle(
            "Lead", parent=base["BodyText"], fontName="Times-Roman", fontSize=11,
            leading=14, spaceAfter=14,
        ),
        "figtitle": ParagraphStyle(
            "FigTitle", parent=base["Heading2"], fontName="Times-Bold", fontSize=12,
            leading=15, spaceBefore=6, spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Times-Roman", fontSize=10,
            leading=13, spaceBefore=6, spaceAfter=4,
        ),
    }


def parse_blocks(markdown: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Return the lead paragraph and a list of (figure title, png name, caption)."""
    lead = ""
    blocks: list[tuple[str, str, str]] = []
    chunks = re.split(r"\n## ", markdown)
    intro = chunks[0]
    for para in intro.split("\n\n"):
        para = para.strip()
        if para and not para.startswith("#"):
            lead = " ".join(para.split())
            break
    for chunk in chunks[1:]:
        head, _, rest = chunk.partition("\n")
        title = head.strip()
        body = " ".join(rest.split()).strip()
        m = FIGURE_RE.search(body)
        if not m:
            continue
        png = m.group(1)
        caption = (body[: m.start()] + body[m.end():]).strip()
        blocks.append((title, png, caption))
    return lead, blocks


def scaled_image(path: Path, max_px: int = 1500) -> Image:
    """Downsample the PNG to a sensible print resolution and embed it as JPEG so
    the PDF stays small."""
    im = PILImage.open(path).convert("RGB")
    if im.width > max_px:
        im = im.resize((max_px, round(im.height * max_px / im.width)), PILImage.LANCZOS)
    buffer = io.BytesIO()
    im.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return Image(buffer, width=CONTENT_WIDTH, height=CONTENT_WIDTH * im.height / im.width)


def main() -> None:
    styles = build_styles()
    lead, blocks = parse_blocks(SOURCE.read_text(encoding="utf-8"))

    story: list = [Paragraph("AtmoResponse figures", styles["title"])]
    if lead:
        story.append(Paragraph(inline_markdown(lead), styles["lead"]))

    for title, png, caption in blocks:
        image_path = HERE / "figures" / png
        parts = [
            Paragraph(inline_markdown(title), styles["figtitle"]),
            scaled_image(image_path),
            Paragraph(inline_markdown(caption), styles["caption"]),
            Spacer(1, 12),
        ]
        story.append(KeepTogether(parts))

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=LETTER, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN, title="AtmoResponse figures",
        author="Paul Peshette",
    )
    document.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
