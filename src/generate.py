import re

import numpy as np

from src.embeddings import model


NOT_FOUND_MESSAGE = "Information not found in the provided notes."

# A sentence must have enough substance to be useful evidence. This removes
# common PDF extraction fragments such as isolated page numbers and labels.
MIN_SENTENCE_WORDS = 4

STOP_WORDS = {
    "what", "why", "how", "when", "where", "which", "who", "whom",
    "is", "are", "was", "were", "the", "a", "an", "of", "to", "in",
    "on", "for", "and", "or", "with", "from", "by", "does", "do",
    "can", "could", "would", "should", "explain", "define", "describe",
    "difference", "between", "give", "list", "mention", "write", "about",
}

QUESTION_HEADING_RE = re.compile(r"^q\s*\d+\s*(?:[-—–:]|\.)?\s*", re.IGNORECASE)

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

    # PDFs often contain tables and headings without sentence punctuation.
    text = re.sub(r"\s+\|\s+", ". ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    cleaned = []
    for sentence in sentences:
        sentence = sentence.strip(" \t\n•-–—")
        if len(re.findall(r"[A-Za-z0-9]+", sentence)) >= MIN_SENTENCE_WORDS:
            cleaned.append(sentence)

    return cleaned


def _clean_source_sentence(sentence):
    """Remove question-paper labels so only answerable source text remains."""
    sentence = sentence.strip()

    if not sentence:
        return None

    if QUESTION_HEADING_RE.match(sentence):
        # A complete question heading should never become an answer bullet.
        question_mark = sentence.find("?")
        if question_mark >= 0:
            remainder = sentence[question_mark + 1:].strip()
            return remainder if remainder else None
        return None

    # Other common question-paper labels should not be treated as evidence.
    lowered = sentence.lower()
    if lowered in {"question paper", "answers", "part a", "part b"}:
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
    """Reward source sentences that explicitly match the requested intent."""
    question_lower = question.lower()
    sentence_words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))

    for intent, terms in INTENT_TERMS.items():
        if intent in question_lower and sentence_words & terms:
            return 0.12

    return 0.0


def _semantic_scores(question, sentences):
    """Score sentences against the question using the local embedding model."""
    if not sentences:
        return []

    embeddings = model.encode(
        [question] + sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    question_vector = np.asarray(embeddings[0], dtype="float32")
    sentence_vectors = np.asarray(embeddings[1:], dtype="float32")

    # With normalized vectors, dot product is cosine similarity.
    return np.dot(sentence_vectors, question_vector).tolist()


def _question_type(question):
    question = question.lower()
    if "difference" in question or "compare" in question or "versus" in question or " vs " in question:
        return "comparison"
    if question.startswith("why "):
        return "why"
    if any(term in question for term in ("characteristic", "characteristics", "feature", "features", "property", "properties")):
        return "characteristics"
    if any(question.startswith(prefix) for prefix in ("list ", "mention ", "name ", "what are ")):
        return "list"
    return "general"


def extract_answer_with_evidence(question, retrieved_chunks, max_sentences=5):
    """Return an extractive answer plus the exact source sentences used."""
    candidates = []
    question_terms = _question_terms(question)
    qtype = _question_type(question)

    for chunk_rank, chunk in enumerate(retrieved_chunks):
        raw_sentences = _split_sentences(chunk["text"])
        sentences = []
        for raw_sentence in raw_sentences:
            cleaned = _clean_source_sentence(raw_sentence)
            if cleaned:
                sentences.append((raw_sentence, cleaned))

        semantic_scores = _semantic_scores(
            question,
            [cleaned for _, cleaned in sentences],
        )

        for sentence_rank, ((raw_sentence, sentence), semantic) in enumerate(
            zip(sentences, semantic_scores)
        ):
            lexical = _lexical_score(sentence, question_terms)

            # Semantic similarity is the primary signal. Lexical overlap helps
            # when a technical term is explicitly present in the source.
            score = 0.70 * float(semantic) + 0.30 * lexical
            score += _intent_score(sentence, question)

            # Prefer stronger FAISS retrieval results without overwhelming the
            # sentence-level semantic score.
            score += max(0.0, 0.015 * (len(retrieved_chunks) - chunk_rank))

            # Comparison questions benefit from sentences containing both
            # compared concepts.
            if qtype == "comparison" and len(question_terms) >= 2:
                sentence_words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                if len(sentence_words & question_terms) >= 2:
                    score += 0.08

            # Characteristics/list questions are better served by several
            # concise source statements than by a generic definition alone.
            if qtype in {"characteristics", "list"}:
                if sentence.strip().startswith(("-", "•")):
                    score += 0.05

            # Avoid weakly related sentences. This threshold is deliberately
            # conservative because unsupported answers must fail closed.
            if score < 0.35:
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

        # For general questions, don't let one page fill the entire answer
        # when another highly relevant page has useful evidence.
        if qtype not in {"comparison", "characteristics", "list"} and page_key in used_pages and len(selected) >= 3:
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


def extract_answer(question, retrieved_chunks, max_sentences=5):
    """Backward-compatible wrapper returning only the answer text."""
    answer, _ = extract_answer_with_evidence(
        question, retrieved_chunks, max_sentences=max_sentences
    )
    return answer


def generate_answer(question, retrieved_chunks):
    """Backward-compatible name for the local extractive answer generator."""
    return extract_answer(question, retrieved_chunks)
