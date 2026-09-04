from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether, HRFlowable

styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle("TitleCustom", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=21, spaceAfter=2 * mm)
SUBTITLE_STYLE = ParagraphStyle("SubtitleCustom", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=9.5, leading=12, textColor=colors.grey, spaceAfter=5 * mm)
QUESTION_STYLE = ParagraphStyle("QuestionCustom", parent=styles["Heading2"], fontSize=12, leading=14.5, spaceBefore=3 * mm, spaceAfter=1.5 * mm, keepWithNext=True)
QUESTION_TEXT_STYLE = ParagraphStyle("QuestionTextCustom", parent=styles["BodyText"], fontSize=9.5, leading=12.5, spaceAfter=2 * mm)
SECTION_STYLE = ParagraphStyle("SectionCustom", parent=styles["Heading3"], fontSize=10.5, leading=13, spaceBefore=1.5 * mm, spaceAfter=1 * mm, keepWithNext=True)
SUBPART_STYLE = ParagraphStyle("SubpartCustom", parent=styles["BodyText"], fontSize=10, leading=13, spaceBefore=1.5 * mm, spaceAfter=1.2 * mm, keepWithNext=True)
BULLET_STYLE = ParagraphStyle("BulletCustom", parent=styles["BodyText"], fontSize=9.5, leading=12.5, leftIndent=5 * mm, firstLineIndent=-3 * mm, spaceAfter=1.1 * mm)
SOURCE_STYLE = ParagraphStyle("SourceCustom", parent=styles["BodyText"], fontSize=8.5, leading=10.5, leftIndent=4 * mm, spaceAfter=0.6 * mm, textColor=colors.HexColor("#444444"))
EVIDENCE_STYLE = ParagraphStyle("EvidenceCustom", parent=styles["BodyText"], fontSize=8.5, leading=10.5, leftIndent=4 * mm, rightIndent=2 * mm, spaceAfter=1.2 * mm)
META_STYLE = ParagraphStyle("MetaCustom", parent=styles["BodyText"], fontSize=8.5, leading=10.5, spaceAfter=1 * mm)


def _escape_text(text):
    if text is None:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clean_display_text(text):
    """Clean OCR/PDF extraction artifacts for presentation only."""
    value = str(text or "")

    # Decorative/OCR glyphs that should never appear in the answer document.
    for glyph in ("■", "▪", "•", "◆", "❖", "❧", "●", "€"):
        value = value.replace(glyph, "")

    # Embedded headers/footers copied from the source notes.
    value = re.sub(r"\b(?:AIMIT/SAP|IT3IPHC504|IT3IPSC521|IT3IPHC)\S*[^.]*?\bPage\s+\d+\b", "", value, flags=re.I)
    value = re.sub(r"\bPage\s+\d+\b", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip(" -:;,.–—")


def _clean_question(text):
    value = _clean_display_text(text)
    value = re.sub(r"\s*\(\s*\d+\s*\)", "", value)
    # Remove leaked exam instructions/section headings from OCR text.
    value = re.sub(r"\s+(?:PART|SECTION)\s*[- ]?[A-ZIV]+\s+Answer any.*$", "", value, flags=re.I)
    return value.strip()


def _format_answer(answer):
    """Turn extracted evidence into a clean exam-style hierarchy."""
    flowables = []
    lines = [x.strip() for x in str(answer or "").splitlines() if x.strip()]
    current_subpart = None

    for raw in lines:
        line = _clean_display_text(raw)
        if not line:
            continue
        line = re.sub(r"^[\-–—*]+\s*", "", line).strip()

        # Answers produced for multipart questions are grouped under a/b/c.
        match = re.match(r"^([a-hA-H])\)\s*(.*)$", line)
        if match:
            label, content = match.groups()
            current_subpart = label.lower()
            flowables.append(Paragraph(f"<b>{current_subpart})</b> {_escape_text(content)}", SUBPART_STYLE))
            continue

        if line.startswith("### ") or line.startswith("## "):
            heading = re.sub(r"^#{2,3}\s*", "", line)
            flowables.append(Paragraph(f"<b>{_escape_text(heading)}</b>", SECTION_STYLE))
            continue

        # Every extracted evidence sentence becomes one visually distinct point.
        flowables.append(Paragraph(f"• {_escape_text(line)}", BULLET_STYLE))

    return flowables


def _page_footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2, 8 * mm, f"Page {document.page}")
    canvas.restoreState()


def _document(output_path, title):
    return SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=13 * mm, bottomMargin=14 * mm,
        title=title, author="RAG Study Assistant",
    )


def _sources_flowables(sources):
    flowables = []
    seen = set()
    for source in sources:
        key = (source.get("file"), source.get("page"))
        if key in seen:
            continue
        seen.add(key)
        filename = Path(str(source.get("file", ""))).name
        flowables.append(Paragraph(
            f"• {_escape_text(filename)} — Page {_escape_text(source.get('page', ''))}", SOURCE_STYLE
        ))
    return flowables


def _question_header(number, question):
    return KeepTogether([
        Paragraph(f"Question {number}", QUESTION_STYLE),
        Paragraph(_escape_text(_clean_question(question)), QUESTION_TEXT_STYLE),
        Paragraph("Answer", SECTION_STYLE),
    ])


def generate_answers_pdf(results, output_path="output/answers.pdf"):
    """Generate a compact, clean, organized exam-answer PDF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = _document(output_path, "RAG Study Assistant - Answers")

    story = [
        Paragraph("RAG Study Assistant", TITLE_STYLE),
        Paragraph("Generated Answers", SUBTITLE_STYLE),
    ]

    for number, result in enumerate(results, start=1):
        story.append(_question_header(number, result.get("question", "")))
        story.extend(_format_answer(result.get("answer", "")))

        sources = result.get("sources", [])
        if sources:
            story.append(Paragraph("Sources", SECTION_STYLE))
            story.extend(_sources_flowables(sources))

        if number < len(results):
            story.append(Spacer(1, 1.5 * mm))
            story.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey, spaceBefore=0.5 * mm, spaceAfter=0.5 * mm))

    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return str(output_path)


def generate_source_mapping_pdf(results, output_path="output/source_mapping.pdf"):
    """Generate a compact source map while cleaning display-only OCR artifacts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = _document(output_path, "RAG Study Assistant - Source Mapping")

    story = [
        Paragraph("RAG Study Assistant", TITLE_STYLE),
        Paragraph("Source Mapping", SUBTITLE_STYLE),
    ]

    for number, result in enumerate(results, start=1):
        story.append(_question_header(number, result.get("question", "")))
        sources = result.get("sources", [])

        if not sources:
            story.append(Paragraph("No source evidence was used.", BODY_STYLE))
        else:
            for source_number, source in enumerate(sources, start=1):
                filename = Path(str(source.get("file", ""))).name
                distance = source.get("distance")
                relevance = source.get("relevance")
                distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else _escape_text(distance)
                relevance_text = f"{relevance:.4f}" if isinstance(relevance, (int, float)) else _escape_text(relevance)

                block = [
                    Paragraph(f"Evidence {source_number}", SECTION_STYLE),
                    Paragraph(
                        f"<b>File:</b> {_escape_text(filename)} &nbsp; | &nbsp; "
                        f"<b>Page:</b> {_escape_text(source.get('page', ''))} &nbsp; | &nbsp; "
                        f"<b>Chunk:</b> {_escape_text(source.get('chunk_id', ''))}<br/>"
                        f"<b>Source type:</b> {_escape_text(source.get('source_type', 'unknown'))} &nbsp; | &nbsp; "
                        f"<b>FAISS distance:</b> {distance_text} &nbsp; | &nbsp; "
                        f"<b>Relevance:</b> {relevance_text}", META_STYLE,
                    ),
                    Paragraph("<b>Exact Answer Evidence</b>", SOURCE_STYLE),
                    Paragraph(_escape_text(_clean_display_text(source.get("text", ""))), EVIDENCE_STYLE),
                ]
                story.append(KeepTogether(block))

        if number < len(results):
            story.append(Spacer(1, 1.5 * mm))
            story.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey, spaceBefore=0.5 * mm, spaceAfter=0.5 * mm))

    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return str(output_path)
