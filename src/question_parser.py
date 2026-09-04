import re


def clean_ocr_text(text):
    """Normalize OCR output while preserving question boundaries."""
    text = str(text or "")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_questions(text):
    """Extract numbered questions from OCR/text input.

    Supports common formats such as:
    Q1. What is an algorithm?
    1. What is an algorithm?
    Q2 — Explain binary search.

    If no numbered questions are found, non-empty lines are treated as
    individual questions so OCR output can still be used.
    """
    text = clean_ocr_text(text)
    if not text:
        return []

    pattern = re.compile(
        r"(?im)(?:^|\n)\s*(?:q(?:uestion)?\s*)?"
        r"(\d{1,3})\s*(?:[.)]|[-—–:]|\s)\s+"
        r"(.+?)(?=\n\s*(?:q(?:uestion)?\s*)?\d{1,3}\s*(?:[.)]|[-—–:]|\s)\s+|\Z)",
        re.DOTALL,
    )

    matches = pattern.findall(text)
    questions = []

    for number, body in matches:
        question = re.sub(r"\s+", " ", body).strip(" -—–")
        question = question.rstrip("|")
        if question and len(question.split()) >= 2:
            questions.append({"number": int(number), "question": question})

    if questions:
        return questions

    # Fallback for OCR text without question numbers.
    fallback = []
    for line in text.splitlines():
        line = line.strip(" •-—–")
        if len(line.split()) >= 3:
            fallback.append({"number": len(fallback) + 1, "question": line})

    return fallback
