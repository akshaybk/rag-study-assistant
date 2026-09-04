import re

import numpy as np

from src.embeddings import model


NOT_FOUND_MESSAGE = "Information not found in the provided notes."
MIN_SENTENCE_WORDS = 4

STOP_WORDS = {
    "what", "why", "how", "when", "where", "which", "who", "whom",
    "is", "are", "was", "were", "the", "a", "an", "of", "to", "in",
    "on", "for", "and", "or", "with", "from", "by", "does", "do",
    "can", "could", "would", "should", "explain", "define", "describe",
    "difference", "between", "give", "list", "mention", "write", "about",
    "briefly", "discuss", "illustrate", "using", "example", "examples",
}

QUESTION_HEADING_RE = re.compile(r"^q\s*\d+\s*(?:[-—–:]|\.)?\s*", re.IGNORECASE)
SUBQUESTION_RE = re.compile(r"(?:^|\s)([a-z])\s*\)\s*", re.IGNORECASE)

INTENT_TERMS = {
    "characteristics": {"characteristic", "characteristics", "feature", "features", "property", "properties"},
    "advantages": {"advantage", "advantages", "benefit", "benefits"},
    "disadvantages": {"disadvantage", "disadvantages", "limitation", "limitations"},
    "applications": {"application", "applications", "use", "uses", "used"},
    "steps": {"step", "steps", "procedure", "process"},
}


def _split_sentences(text):
    """Split source text into useful sentence-like units."""
    text = " ".join(str(text).split())
    if not text:
        return []

    text = re.sub(r"\s+\|\s+", ". ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    cleaned = []
    for sentence in sentences:
        sentence = sentence.strip(" \t\n•-–—")
        if len(re.findall(r"[A-Za-z0-9]+", sentence)) >= MIN_SENTENCE_WORDS:
            cleaned.append(sentence)
    return cleaned


def _clean_source_sentence(sentence):
    """Remove question-paper labels and metadata from answer evidence."""
    sentence = sentence.strip()
    if not sentence:
        return None

    if QUESTION_HEADING_RE.match(sentence):
        question_mark = sentence.find("?")
        if question_mark >= 0:
            remainder = sentence[question_mark + 1:].strip()
            return remainder if remainder else None
        return None

    lowered = sentence.lower()
    if lowered in {"question paper", "answers", "part a", "part b", "section - 1", "section - iv", "section -v"}:
        return None

    if any(marker in lowered for marker in (
        "reg. no.", "semester -", "examination -", "max. marks", "time:",
    )):
        return None

    return sentence


def _question_terms(question):
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    return {word for word in words if word not in STOP_WORDS and len(word) > 2}


def _lexical_score(sentence, question_terms):
    words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
    if not words or not question_terms:
        return 0.0
    return len(words & question_terms) / len(question_terms)


def _intent_score(sentence, question):
    question_lower = question.lower()
    sentence_words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
    for intent, terms in INTENT_TERMS.items():
        if intent in question_lower and sentence_words & terms:
            return 0.12
    return 0.0


def _semantic_scores(question, sentences):
    if not sentences:
        return []
    embeddings = model.encode(
        [question] + sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    question_vector = np.asarray(embeddings[0], dtype="float32")
    sentence_vectors = np.asarray(embeddings[1:], dtype="float32")
    return np.dot(sentence_vectors, question_vector).tolist()


def _question_type(question):
    question = question.lower()
    if "difference" in question or "compare" in question or "versus" in question or " vs " in question:
        return "comparison"
    if question.startswith("why "):
        return "why"
    if any(term in question for term in ("characteristic", "characteristics", "feature", "features", "property", "properties")):
        return "characteristics"
    if any(term in question for term in ("advantage", "advantages", "benefit", "benefits")):
        return "advantages"
    if any(term in question for term in ("disadvantage", "disadvantages", "limitation", "limitations")):
        return "disadvantages"
    if any(question.startswith(prefix) for prefix in ("list ", "mention ", "name ", "what are ")):
        return "list"
    return "general"


def split_subquestions(question):
    """Split an exam question containing a), b), ... into independent parts."""
    text = " ".join(str(question).split())
    matches = list(SUBQUESTION_RE.finditer(text))
    if not matches:
        return [(None, text)]

    parts = []
    prefix = text[:matches[0].start()].strip()
    if prefix:
        parts.append((None, prefix))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        part_text = text[start:end].strip()
        part_text = re.sub(
            r"^(?:\(?\d+\)?\s*(?:marks?)?\s*)",
            "",
            part_text,
            flags=re.IGNORECASE,
        )
        if part_text:
            parts.append((match.group(1).lower(), part_text))

    return parts or [(None, text)]


def _extract_single_answer(question, retrieved_chunks, max_sentences=4):
    """Extract evidence for one question or sub-question."""
    candidates = []
    question_terms = _question_terms(question)
    qtype = _question_type(question)

    for chunk_rank, chunk in enumerate(retrieved_chunks):
        # A previous question paper is not authoritative evidence for a new answer.
        if chunk.get("source_type") == "previous_question_paper":
            continue

        raw_sentences = _split_sentences(chunk["text"])
        sentences = []
        for raw_sentence in raw_sentences:
            cleaned = _clean_source_sentence(raw_sentence)
            if cleaned:
                sentences.append((raw_sentence, cleaned))

        semantic_scores = _semantic_scores(question, [cleaned for _, cleaned in sentences])

        for sentence_rank, ((raw_sentence, sentence), semantic) in enumerate(zip(sentences, semantic_scores)):
            lexical = _lexical_score(sentence, question_terms)
            score = 0.55 * float(semantic) + 0.45 * lexical
            score += _intent_score(sentence, question)
            score += max(0.0, 0.015 * (len(retrieved_chunks) - chunk_rank))

            sentence_words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
            matched_terms = sentence_words & question_terms

            if qtype == "comparison" and len(matched_terms) >= 2:
                score += 0.10

            if qtype in {"characteristics", "list", "advantages", "disadvantages"} and matched_terms:
                score += 0.03

            # Semantic similarity alone is not enough for a technical answer.
            # Require a meaningful lexical anchor unless similarity is very high.
            if not matched_terms and float(semantic) < 0.72:
                continue
            if score < 0.42:
                continue

            candidates.append({
                "sentence": sentence,
                "score": score,
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "source": chunk["source"],
                "source_type": chunk.get("source_type", "unknown"),
                "sentence_rank": sentence_rank,
                "raw_sentence": raw_sentence,
            })

    if not candidates:
        return NOT_FOUND_MESSAGE, []

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    seen = set()
    used_pages = set()

    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate["sentence"].lower()).strip()
        page_key = (candidate["source"], candidate["page"])
        if normalized in seen:
            continue

        if qtype not in {"comparison", "characteristics", "list", "advantages", "disadvantages"} and page_key in used_pages and len(selected) >= 3:
            continue

        selected.append(candidate)
        seen.add(normalized)
        used_pages.add(page_key)
        if len(selected) >= max_sentences:
            break

    if not selected:
        return NOT_FOUND_MESSAGE, []

    selected.sort(key=lambda item: (item["chunk_id"], item["sentence_rank"]))
    answer = "\n".join(f"- {item['sentence']}" for item in selected)
    return answer, selected


def extract_answer_with_evidence(question, retrieved_chunks, max_sentences=4):
    """Return an extractive answer plus exact evidence used.

    Multi-part exam questions are answered independently so evidence for part
    (a) cannot contaminate part (b).
    """
    parts = split_subquestions(question)
    if len(parts) == 1:
        return _extract_single_answer(question, retrieved_chunks, max_sentences)

    answer_parts = []
    all_evidence = []

    for label, subquestion in parts:
        answer, evidence = _extract_single_answer(
            subquestion,
            retrieved_chunks,
            max_sentences=max_sentences,
        )
        answer_parts.append(f"{label}) {answer}" if label else answer)
        all_evidence.extend(evidence)

    if not all_evidence:
        return NOT_FOUND_MESSAGE, []

    return "\n\n".join(answer_parts), all_evidence


def extract_answer(question, retrieved_chunks, max_sentences=4):
    """Backward-compatible wrapper returning only the answer text."""
    answer, _ = extract_answer_with_evidence(
        question, retrieved_chunks, max_sentences=max_sentences
    )
    return answer


def generate_answer(question, retrieved_chunks):
    """Backward-compatible name for the local extractive answer generator."""
    return extract_answer(question, retrieved_chunks)
