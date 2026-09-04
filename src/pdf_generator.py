from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    KeepTogether,
)


# =========================
# STYLES
# =========================

styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "TitleCustom",
    parent=styles["Title"],
    alignment=TA_CENTER,
    spaceAfter=10 * mm,
)

QUESTION_STYLE = ParagraphStyle(
    "QuestionCustom",
    parent=styles["Heading2"],
    spaceBefore=4 * mm,
    spaceAfter=3 * mm,
)

BODY_STYLE = ParagraphStyle(
    "BodyCustom",
    parent=styles["BodyText"],
    leading=15,
    spaceAfter=4 * mm,
)

SOURCE_STYLE = ParagraphStyle(
    "SourceCustom",
    parent=styles["BodyText"],
    leading=13,
    spaceAfter=2 * mm,
)

EVIDENCE_STYLE = ParagraphStyle(
    "EvidenceCustom",
    parent=styles["BodyText"],
    leftIndent=5 * mm,
    rightIndent=5 * mm,
    leading=13,
    spaceAfter=4 * mm,
)


# =========================
# HELPERS
# =========================

def _escape_text(text):
    """Escape text so it can safely be used in a ReportLab Paragraph."""
    if text is None:
        return ""

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_answer(answer):
    """Convert simple Markdown-style answer text into PDF paragraphs."""
    flowables = []

    for raw_line in str(answer).splitlines():
        line = raw_line.strip()

        if not line:
            flowables.append(Spacer(1, 2 * mm))
            continue

        # Markdown headings
        if line.startswith("### "):
            heading = _escape_text(line[4:])
            flowables.append(Paragraph(f"<b>{heading}</b>", QUESTION_STYLE))
            continue

        if line.startswith("## "):
            heading = _escape_text(line[3:])
            flowables.append(Paragraph(f"<b>{heading}</b>", QUESTION_STYLE))
            continue

        # Markdown bullets
        if line.startswith("- ") or line.startswith("* "):
            bullet = _escape_text(line[2:])
            flowables.append(Paragraph(f"• {bullet}", BODY_STYLE))
            continue

        # Numbered list items
        if len(line) > 2 and line[0].isdigit() and line[1:3] == ". ":
            flowables.append(Paragraph(_escape_text(line), BODY_STYLE))
            continue

        flowables.append(Paragraph(_escape_text(line), BODY_STYLE))

    return flowables


# =========================
# ANSWERS PDF
# =========================

def generate_answers_pdf(results, output_path="output/answers.pdf"):
    """
    Generate an exam-ready Answers PDF from RAG results.

    Each result should contain:
        question
        answer
        sources
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="RAG Study Assistant - Answers",
    )

    story = [
        Paragraph("RAG Study Assistant", TITLE_STYLE),
        Paragraph("Generated Answers", TITLE_STYLE),
    ]

    for number, result in enumerate(results, start=1):
        question = _escape_text(result.get("question", ""))

        question_block = [
            Paragraph(f"Question {number}", QUESTION_STYLE),
            Paragraph(question, BODY_STYLE),
            Paragraph("Answer", QUESTION_STYLE),
        ]

        question_block.extend(_format_answer(result.get("answer", "")))

        # Show unique source file/page combinations.
        source_keys = set()
        source_lines = []

        for source in result.get("sources", []):
            key = (source.get("file"), source.get("page"))
            if key in source_keys:
                continue

            source_keys.add(key)
            filename = Path(str(source.get("file", ""))).name
            source_lines.append(
                f"• {_escape_text(filename)} — Page {source.get('page', '')}"
            )

        if source_lines:
            question_block.append(Paragraph("Sources", QUESTION_STYLE))
            for line in source_lines:
                question_block.append(Paragraph(line, SOURCE_STYLE))

        story.extend(question_block)
        story.append(PageBreak())

    # Remove the final unnecessary page break.
    if story and isinstance(story[-1], PageBreak):
        story.pop()

    document.build(story)

    return str(output_path)


# =========================
# SOURCE MAPPING PDF
# =========================

def generate_source_mapping_pdf(results, output_path="output/source_mapping.pdf"):
    """
    Generate a Source Mapping PDF showing the retrieved evidence
    used for each generated answer.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="RAG Study Assistant - Source Mapping",
    )

    story = [
        Paragraph("RAG Study Assistant", TITLE_STYLE),
        Paragraph("Source Mapping", TITLE_STYLE),
    ]

    for number, result in enumerate(results, start=1):
        question = _escape_text(result.get("question", ""))

        story.append(Paragraph(f"Question {number}", QUESTION_STYLE))
        story.append(Paragraph(question, BODY_STYLE))

        sources = result.get("sources", [])

        if not sources:
            story.append(
                Paragraph(
                    "No source evidence was retrieved.",
                    BODY_STYLE,
                )
            )
            continue

        for source_number, source in enumerate(sources, start=1):
            filename = Path(str(source.get("file", ""))).name
            page = source.get("page", "")
            chunk_id = source.get("chunk_id", "")
            distance = source.get("distance")
            evidence = _escape_text(source.get("text", ""))

            if isinstance(distance, (int, float)):
                distance_text = f"{distance:.4f}"
            else:
                distance_text = _escape_text(distance)

            block = [
                Paragraph(f"<b>Source {source_number}</b>", QUESTION_STYLE),
                Paragraph(
                    f"<b>File:</b> {_escape_text(filename)}<br/>"
                    f"<b>Page:</b> {_escape_text(page)}<br/>"
                    f"<b>Chunk:</b> {_escape_text(chunk_id)}<br/>"
                    f"<b>FAISS distance:</b> {distance_text}",
                    SOURCE_STYLE,
                ),
                Paragraph("<b>Retrieved Evidence</b>", SOURCE_STYLE),
                Paragraph(evidence, EVIDENCE_STYLE),
            ]

            story.append(KeepTogether(block))

        story.append(PageBreak())

    if story and isinstance(story[-1], PageBreak):
        story.pop()

    document.build(story)

    return str(output_path)
