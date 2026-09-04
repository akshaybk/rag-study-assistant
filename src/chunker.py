import re


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


def _split_units(text):
    """Split PDF text into sentence/line-sized units without losing headings."""
    units = []

    for line in str(text).splitlines():
        line = " ".join(line.split()).strip()
        if not line:
            continue

        # Keep normal sentences together, but also handle PDF tables/headings
        # that may have little or no punctuation.
        parts = re.split(r"(?<=[.!?])\s+", line)
        units.extend(part.strip() for part in parts if part.strip())

    return units


def _hard_split(text, chunk_size):
    """Fallback for a single unit longer than the configured chunk size."""
    return [text[start:start + chunk_size] for start in range(0, len(text), chunk_size)]


def chunk_pages(pages, chunk_size=500, overlap=100):
    """
    Split extracted PDF pages into overlapping chunks while preferring
    sentence/line boundaries.

    Each chunk keeps its source, page number, and source type. This avoids
    cutting an answer in the middle of a sentence whenever possible.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    chunk_id = 0

    for page in pages:
        source_type = _detect_source_type(page["text"])
        units = _split_units(page["text"])
        start = 0

        while start < len(units):
            # A single oversized unit (usually a long extracted table row)
            # cannot respect the normal boundary-based chunking strategy.
            if len(units[start]) > chunk_size:
                for piece in _hard_split(units[start], chunk_size):
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": piece,
                        "page": page["page"],
                        "source": page["source"],
                        "source_type": source_type,
                    })
                    chunk_id += 1
                start += 1
                continue

            current = []
            current_length = 0
            cursor = start

            while cursor < len(units):
                unit = units[cursor]
                separator_length = 1 if current else 0
                proposed_length = current_length + separator_length + len(unit)

                if current and proposed_length > chunk_size:
                    break

                current.append(unit)
                current_length = proposed_length
                cursor += 1

            chunk_text = " ".join(current)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"],
                "source_type": source_type,
            })
            chunk_id += 1

            if cursor >= len(units):
                break

            # Retain enough previous units to approximate the requested overlap.
            overlap_length = 0
            next_start = cursor
            for index in range(cursor - 1, start - 1, -1):
                unit_length = len(units[index]) + (1 if overlap_length else 0)
                if overlap_length + unit_length > overlap:
                    break
                overlap_length += unit_length
                next_start = index

            start = next_start if next_start < cursor else cursor

    return chunks
