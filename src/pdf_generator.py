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

        if line.startswith("### "):
            flowables.append(
                Paragraph(f"<b>{_escape_text(line[4:])}</b>", QUESTION_STYLE)
            )
            continue

        if line.startswith("## "):
            flowables.append(
                Paragraph(f"<b>{_escape_text(line[3:])}</b>", QUESTION_STYLE)
            )
            continue

        if line.startswith("- ") or line.startswith("* "):
            flowables.append(Paragraph(_escape_text(line), BODY_STYLE))
            continue

        flowables.append(Paragraph(_escape_text(line), BODY_STYLE))

    return flowables


def generate_answers_pdf(results, output_path="output/answers.pdf"):
    """Generate an exam-ready Answers PDF from RAG results."""
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
        question_block = [
            Paragraph(f"Question {number}", QUESTION_STYLE),
            Paragraph(_escape_text(result.get("question", "")), BODY_STYLE),
            Paragraph("Answer", QUESTION_STYLE),
        ]
        question_block.extend(_format_answer(result.get("answer", "")))

        source_keys = set()
        source_lines = []
        for source in result.get("sources", []):
            key = (source.get("file"), source.get("page"))
            if key in source_keys:
                continue
            source_keys.add(key)
            filename = Path(str(source.get("file", ""))).name
            source_lines.append(
                f"- {_escape_text(filename)} — Page {source.get('page', '')}"
            )

        if source_lines:
            question_block.append(Paragraph("Sources", QUESTION_STYLE))
            for line in source_lines:
                question_block.append(Paragraph(line, SOURCE_STYLE))

        story.extend(question_block)
        story.append(PageBreak())

    if story and isinstance(story[-1], PageBreak):
        story.pop()

    document.build(story)
    return str(output_path)


def generate_source_mapping_pdf(results, output_path="output/source_mapping.pdf"):
    """Generate a Source Mapping PDF showing exact answer evidence."""
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
        story.append(Paragraph(f"Question {number}", QUESTION_STYLE))
        story.append(Paragraph(_escape_text(result.get("question", "")), BODY_STYLE))

        sources = result.get("sources", [])
        if not sources:
            story.append(Paragraph("No source evidence was used.", BODY_STYLE))
            story.append(PageBreak())
            continue

        for source_number, source in enumerate(sources, start=1):
            filename = Path(str(source.get("file", ""))).name
            page = source.get("page", "")
            chunk_id = source.get("chunk_id", "")
            source_type = source.get("source_type", "unknown")
            distance = source.get("distance")
            relevance = source.get("relevance")
            evidence = source.get("text", "")

            distance_text = (
                f"{distance:.4f}" if isinstance(distance, (int, float))
                else _escape_text(distance)
            )
            relevance_text = (
                f"{relevance:.4f}" if isinstance(relevance, (int, float))
                else _escape_text(relevance)
            )

            block = [
                Paragraph(f"<b>Evidence {source_number}</b>", QUESTION_STYLE),
                Paragraph(
                    f"<b>File:</b> {_escape_text(filename)}<br/>"
                    f"<b>Source type:</b> {_escape_text(source_type)}<br/>"
                    f"<b>Page:</b> {_escape_text(page)}<br/>"
                    f"<b>Chunk:</b> {_escape_text(chunk_id)}<br/>"
                    f"<b>FAISS distance:</b> {distance_text}<br/>"
                    f"<b>Relevance:</b> {relevance_text}",
                    SOURCE_STYLE,
                ),
                Paragraph("<b>Exact Answer Evidence</b>", SOURCE_STYLE),
                Paragraph(_escape_text(evidence), EVIDENCE_STYLE),
            ]
            story.append(KeepTogether(block))

        story.append(PageBreak())

    if story and isinstance(story[-1], PageBreak):
        story.pop()

    document.build(story)
    return str(output_path)
