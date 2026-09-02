"""Render PROJECT_SUMMARY.md as a compact fixed-layout PDF.

The committed PROJECT_SUMMARY.pdf is produced by this script. Regenerate it after
editing PROJECT_SUMMARY.md:

    pip install reportlab
    python report/render_pdf.py

Uses reportlab's built-in Times family, so it needs no system fonts and runs on
any platform.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "PROJECT_SUMMARY.md"
OUTPUT = HERE / "PROJECT_SUMMARY.pdf"
MARGIN = 0.75 * inch

BODY = "Times-Roman"
BOLD = "Times-Bold"
ITALIC = "Times-Italic"


def inline_markdown(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return re.sub(r"(?<!\*)\*([^*]+)\*", r"<i>\1</i>", escaped)


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "MemoTitle",
            parent=styles["Title"],
            fontName=BOLD,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "tagline": ParagraphStyle(
            "Tagline",
            parent=styles["BodyText"],
            fontName=ITALIC,
            fontSize=11,
            leading=13,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName=BODY,
            fontSize=11,
            leading=13,
            spaceAfter=7,
        ),
        "summary": ParagraphStyle(
            "Summary",
            parent=styles["BodyText"],
            fontName=BODY,
            fontSize=11,
            leading=13,
            spaceAfter=10,
            borderColor=colors.HexColor("#808080"),
            borderWidth=0.75,
            borderPadding=8,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=styles["Heading2"],
            fontName=BOLD,
            fontSize=14,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            parent=styles["Heading3"],
            fontName=BOLD,
            fontSize=12,
            leading=14,
            spaceBefore=7,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=styles["BodyText"],
            fontName=BODY,
            fontSize=10,
            leading=12,
            leftIndent=16,
            firstLineIndent=-16,
            spaceAfter=6,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=styles["BodyText"],
            fontName=BODY,
            fontSize=10,
            leading=12,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=styles["BodyText"],
            fontName=BOLD,
            fontSize=10,
            leading=12,
        ),
    }


def make_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)

    width = LETTER[0] - 2 * MARGIN
    col_widths = [width * fraction for fraction in (0.24, 0.22, 0.32, 0.22)]
    if len(rows[0]) != 4:
        col_widths = [width / len(rows[0])] * len(rows[0])

    rendered = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        rendered.append([Paragraph(inline_markdown(cell), style) for cell in row])

    table = Table(rendered, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#909090")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def page_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont(BODY, 9)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(LETTER[0] / 2, 0.5 * inch, f"AtmoResponse | Page {document.page}")
    canvas.restoreState()


def parse_story(markdown: str, styles: dict[str, ParagraphStyle]) -> list:
    clean = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL).strip()
    lines = clean.splitlines()
    story = []
    index = 0
    before_first_section = True

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(Spacer(1, 3))
            story.append(make_table(table_lines, styles))
            story.append(Spacer(1, 9))
            continue
        if line == "---":
            story.append(Spacer(1, 5))
            story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#808080")))
            story.append(Spacer(1, 5))
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markdown(line[2:]), styles["title"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markdown(line[3:]), styles["h2"]))
            before_first_section = False
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markdown(line[4:]), styles["h3"]))
            index += 1
            continue

        paragraph_lines = []
        is_reference = line.startswith("- ")
        if is_reference:
            paragraph_lines.append(line[2:])
        else:
            paragraph_lines.append(line)
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if not next_line or next_line.startswith(("#", "|")) or next_line == "---":
                break
            if next_line.startswith("- "):
                break
            paragraph_lines.append(next_line)
            index += 1

        text = " ".join(paragraph_lines)
        if is_reference:
            story.append(Paragraph(f"- {inline_markdown(text)}", styles["reference"]))
        elif before_first_section and text.startswith("*") and text.endswith("*"):
            story.append(Paragraph(inline_markdown(text), styles["tagline"]))
        elif before_first_section:
            story.append(Paragraph(inline_markdown(text), styles["summary"]))
        else:
            story.append(Paragraph(inline_markdown(text), styles["body"]))
    return story


def main() -> None:
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="AtmoResponse",
        author="Paul Peshette",
    )
    story = parse_story(SOURCE.read_text(encoding="utf-8"), build_styles())
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
