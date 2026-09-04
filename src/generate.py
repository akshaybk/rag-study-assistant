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
    "briefly", "following", "given", "using", "example", "examples",
}

QUESTION_HEADING_RE = re.compile(r"^q\s*\d+\s*(?:[-—–:]|\.)?\s*", re.IGNORECASE)
SUBQUESTION_RE = re.compile(r"(?:^|\s)([a-hA-H])\s*[\)\.]\s*")

INTENT_TERMS = {
    "characteristics": {"characteristic", "characteristics", "feature", "features", "property", "properties"},
    "advantages": {"advantage", "advantages", "benefit", "benefits"},
    "disadvantages": {"disadvantage", "disadvantages", "limitation", "limitations"},
    "applications": {"application", "applications", "use", "uses", "used"},
    "steps": {"step", "steps", "procedure", "process"},
    "methods": {"method", "methods"},
}


def split_subquestions(question):
    """Split a question containing a), b), ... into independent questions."""
    text = " ".join(str(question).split())
    matches = list(SUBQUESTION_RE.finditer(text))
    if not matches:
        return [("", text)] if text else []

    parts = []
    for index, match in enumerate(matches):
        label = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        part = text[start:end].strip()
        part = re.sub(r"\(\s*\d+\s*\)", "", part).strip()
        if part:
            parts.append((label, part))
    return parts


def _split_sentences(text):
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
    sentence = sentence.strip()
    if not sentence:
        return None
    if QUESTION_HEADING_RE.match(sentence):
        question_mark = sentence.find("?")
        if question_mark >= 0:
            remainder = sentence[question_mark + 1:].strip()
            return remainder if remainder else None
        return None
    lowered = sentence.lower().strip(" .:-")
    if lowered in {"question paper", "answers", "part a", "part b", "part c", "section i", "section ii", "section iii", "section iv", "section v"}:
        return None
    if lowered.startswith("references:") or lowered.startswith("reference:"):
        return None
    return sentence


def _question_terms(question):
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    return {word for word in words if word not in STOP_WORDS and len(word) > 2}


def _lexical_score(sentence, question_terms):
    words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
    if not words or not question_terms:
        return 0.0
    exact = len(words & question_terms) / len(question_terms)
    stemmed = sum(
        1 for term in question_terms
        if any(word.startswith(term[:5]) for word in words if len(term) >= 5)
    ) / len(question_terms)
    return max(exact, 0.8 * stemmed)


def _intent_score(sentence, question):
    question_lower = question.lower()
    sentence_words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
    for intent, terms in INTENT_TERMS.items():
        if intent in question_lower and sentence_words & terms:
            return 0.12
    if "four methods" in question_lower and sentence_words & INTENT_TERMS["methods"]:
        return 0.10
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
    q = question.lower()
    if "difference" in q or "compare" in q or "versus" in q or " vs " in q:
        return "comparison"
    if q.startswith("why "):
        return "why"
    if any(t in q for t in ("characteristic", "feature", "property")):
        return "characteristics"
    if "four methods" in q or any(q.startswith(p) for p in ("list ", "mention ", "name ", "what are ")):
        return "list"
    return "general"


def _extract_single(question, retrieved_chunks, max_sentences=5):
    candidates = []
    question_terms = _question_terms(question)
    qtype = _question_type(question)

    # Previous question papers are retrieval material, not permitted evidence.
    answer_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk.get("source_type", "study_notes") != "previous_question_paper"
    ]

    for chunk_rank, chunk in enumerate(answer_chunks):
        pairs = []
        for raw in _split_sentences(chunk["text"]):
            cleaned = _clean_source_sentence(raw)
            if cleaned:
                pairs.append((raw, cleaned))

        semantic_scores = _semantic_scores(question, [cleaned for _, cleaned in pairs])

        for sentence_rank, ((raw_sentence, sentence), semantic) in enumerate(zip(pairs, semantic_scores)):
            lexical = _lexical_score(sentence, question_terms)
            score = 0.45 * float(semantic) + 0.55 * lexical
            score += _intent_score(sentence, question)
            score += max(0.0, 0.015 * (len(answer_chunks) - chunk_rank))

            if qtype == "comparison" and len(question_terms) >= 2:
                words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                if len(words & question_terms) >= 2:
                    score += 0.12

            # A sentence must either contain a meaningful question term or be
            # exceptionally semantically close. This blocks unrelated prose.
            if lexical == 0.0 and semantic < 0.72:
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


def extract_answer_with_evidence(question, retrieved_chunks, max_sentences=5):
    """Answer each subquestion independently using only note evidence."""
    parts = split_subquestions(question)
    if len(parts) <= 1:
        return _extract_single(question, retrieved_chunks, max_sentences=max_sentences)

    answers = []
    all_evidence = []
    for label, subquestion in parts:
        answer, evidence = _extract_single(subquestion, retrieved_chunks, max_sentences=max_sentences)
        prefix = f"{label}) " if label else ""
        if answer == NOT_FOUND_MESSAGE:
            answers.append(f"{prefix}{NOT_FOUND_MESSAGE}")
        else:
            lines = answer.splitlines()
            if lines:
                answers.append(f"{prefix}{lines[0][2:] if lines[0].startswith('- ') else lines[0]}")
                answers.extend(lines[1:])
        all_evidence.extend(evidence)

    if not all_evidence:
        return NOT_FOUND_MESSAGE, []
    return "\n".join(answers), all_evidence


def extract_answer(question, retrieved_chunks, max_sentences=5):
    answer, _ = extract_answer_with_evidence(question, retrieved_chunks, max_sentences=max_sentences)
    return answer


def generate_answer(question, retrieved_chunks):
    return extract_answer(question, retrieved_chunks)
