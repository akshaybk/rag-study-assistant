def _detect_source_type(text):
    """Classify a page so retrieval results retain basic source information."""

    normalized = text.lower()

    question_paper_markers = [
        "question paper",
        "question paper answers",
        "part a — 2-mark answers",
        "part a - 2-mark answers",
        "q1 —",
        "q1 -",
    ]

    if any(marker in normalized for marker in question_paper_markers):
        return "previous_question_paper"

    return "study_notes"


def chunk_pages(pages, chunk_size=500, overlap=100):
    """
    Split extracted PDF pages into smaller overlapping chunks.

    Each chunk keeps its source, page number, and source type.
    """

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        text = " ".join(text.split())
        source_type = _detect_source_type(text)

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"],
                "source_type": source_type,
            })

            chunk_id += 1
            start += chunk_size - overlap

    return chunks
