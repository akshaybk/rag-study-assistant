from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    KeepTogether,
    Table,
    TableStyle,
)


styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "TitleCustom",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=18,
    leading=22,
    spaceAfter=2 * mm,
)

SUBTITLE_STYLE = ParagraphStyle(
    "SubtitleCustom",
    parent=styles["BodyText"],
    alignment=TA_CENTER,
    fontSize=9.5,
    leading=12,
    textColor=colors.grey,
    spaceAfter=7 * mm,
)

QUESTION_STYLE = ParagraphStyle(
    "QuestionCustom",
    parent=styles["Heading2"],
    fontSize=11.5,
    leading=14,
    spaceBefore=3 * mm,
    spaceAfter=1.5 * mm,
    keepWithNext=True,
)

SECTION_STYLE = ParagraphStyle(
    "SectionCustom",
    parent=styles["Heading3"],
    fontSize=10,
    leading=12,
    spaceBefore=2 * mm,
    spaceAfter=1 * mm,
    keepWithNext=True,
)

QUESTION_TEXT_STYLE = ParagraphStyle(
    "QuestionTextCustom",
    parent=styles["BodyText"],
    fontSize=9.5,
    leading=12.5,
    spaceAfter=2.5 * mm,
)

BODY_STYLE = ParagraphStyle(
    "BodyCustom",
    parent=styles["BodyText"],
    fontSize=9.5,
    leading=13,
    spaceAfter=1.8 * mm,
)

BULLET_STYLE = ParagraphStyle(
    "BulletCustom",
    parent=BODY_STYLE,
    leftIndent=5 * mm,
    firstLineIndent=-3 * mm,
    spaceAfter=1.2 * mm,
)

SOURCE_STYLE = ParagraphStyle(
    "SourceCustom",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=11,
    leftIndent=3 * mm,
    spaceAfter=0.8 * mm,
)

EVIDENCE_STYLE = ParagraphStyle(
    "EvidenceCustom",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=11,
    leftIndent=4 * mm,
    rightIndent=2 * mm,
    spaceAfter=1.5 * mm,
)

MAPPING_META_STYLE = ParagraphStyle(
    "MappingMetaCustom",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=11,
    spaceAfter=1.5 * mm,
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
    """Convert simple Markdown-style answer text into compact PDF flowables."""
    flowables = []

    for raw_line in str(answer).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("### "):
            flowables.append(Paragraph(_escape_text(line[4:]), SECTION_STYLE))
            continue

        if line.startswith("## "):
            flowables.append(Paragraph(_escape_text(line[3:]), SECTION_STYLE))
            continue

        if line.startswith("- ") or line.startswith("* "):
            text = _escape_text(line[2:].strip())
            flowables.append(Paragraph(f"• {text}", BULLET_STYLE))
            continue

        flowables.append(Paragraph(_escape_text(line), BODY_STYLE))

    return flowables


def _page_footer(canvas, document):
    """Draw a small page number without consuming layout space."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2, 8 * mm, f"Page {document.page}")
    canvas.restoreState()


def _document(output_path, title):
    return SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=title,
        author="RAG Study Assistant",
    )


def _sources_flowables(sources):
    """Create a compact source list, de-duplicated by file and page."""
    source_keys = set()
    flowables = []

    for source in sources:
        key = (source.get("file"), source.get("page"))
        if key in source_keys:
            continue
        source_keys.add(key)
        filename = Path(str(source.get("file", ""))).name
        flowables.append(
            Paragraph(
                f"• {_escape_text(filename)} — Page {_escape_text(source.get('page', ''))}",
                SOURCE_STYLE,
            )
        )

    return flowables


def generate_answers_pdf(results, output_path="output/answers.pdf"):
    """Generate a compact, exam-ready Answers PDF from RAG results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = _document(output_path, "RAG Study Assistant - Answers")

    story = [
        Paragraph("RAG Study Assistant", TITLE_STYLE),
        Paragraph("Generated Answers", SUBTITLE_STYLE),
    ]

    for number, result in enumerate(results, start=1):
        question = result.get("question", "")
        answer = result.get("answer", "")

        # Keep only the question heading and question text together. The answer
        # is allowed to flow naturally across pages instead of forcing a whole
        # question onto one page.
        story.append(
            KeepTogether([
                Paragraph(f"Question {number}", QUESTION_STYLE),
                Paragraph(_escape_text(question), QUESTION_TEXT_STYLE),
                Paragraph("Answer", SECTION_STYLE),
            ])
        )
        story.extend(_format_answer(answer))

        sources = result.get("sources", [])
        if sources:
            story.append(Paragraph("Sources", SECTION_STYLE))
            story.extend(_sources_flowables(sources))

        # Small separator instead of a page break. This removes the large blank
        # areas caused by forcing every question onto a new page.
        story.append(Spacer(1, 2.5 * mm))
        story.append(
            Table(
                [[""]],
                colWidths=[178 * mm],
                rowHeights=[0.25 * mm],
                style=TableStyle([
                    ("LINEBELOW", (0, 0), (-1, -1), 0.35, colors.lightgrey),
                ]),
            )
        )

    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return str(output_path)


def generate_source_mapping_pdf(results, output_path="output/source_mapping.pdf"):
    """Generate a compact Source Mapping PDF showing exact answer evidence."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = _document(output_path, "RAG Study Assistant - Source Mapping")

    story = [
        Paragraph("RAG Study Assistant", TITLE_STYLE),
        Paragraph("Source Mapping", SUBTITLE_STYLE),
    ]

    for number, result in enumerate(results, start=1):
        story.append(Paragraph(f"Question {number}", QUESTION_STYLE))
        story.append(Paragraph(_escape_text(result.get("question", "")), QUESTION_TEXT_STYLE))

        sources = result.get("sources", [])
        if not sources:
            story.append(Paragraph("No source evidence was used.", BODY_STYLE))
            story.append(Spacer(1, 2 * mm))
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
                f"{distance:.4f}" if isinstance(distance, (int, float)) else _escape_text(distance)
            )
            relevance_text = (
                f"{relevance:.4f}" if isinstance(relevance, (int, float)) else _escape_text(relevance)
            )

            meta = (
                f"<b>File:</b> {_escape_text(filename)} &nbsp; | &nbsp; "
                f"<b>Page:</b> {_escape_text(page)} &nbsp; | &nbsp; "
                f"<b>Chunk:</b> {_escape_text(chunk_id)}<br/>"
                f"<b>Source type:</b> {_escape_text(source_type)} &nbsp; | &nbsp; "
                f"<b>FAISS distance:</b> {distance_text} &nbsp; | &nbsp; "
                f"<b>Relevance:</b> {relevance_text}"
            )

            block = [
                Paragraph(f"Evidence {source_number}", SECTION_STYLE),
                Paragraph(meta, MAPPING_META_STYLE),
                Paragraph("<b>Exact Answer Evidence</b>", MAPPING_META_STYLE),
                Paragraph(_escape_text(evidence), EVIDENCE_STYLE),
            ]
            story.append(KeepTogether(block))

        story.append(Spacer(1, 2.5 * mm))
        story.append(
            Table(
                [[""]],
                colWidths=[178 * mm],
                rowHeights=[0.25 * mm],
                style=TableStyle([
                    ("LINEBELOW", (0, 0), (-1, -1), 0.35, colors.lightgrey),
                ]),
            )
        )

    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return str(output_path)
