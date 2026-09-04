import re


NOT_FOUND_MESSAGE = "Information not found in the provided notes."


def _split_sentences(text):
    """Split source text into readable sentences without changing their wording."""
    text = " ".join(str(text).split())

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _question_terms(question):
    """Return useful question terms for deterministic sentence scoring."""
    stop_words = {
        "what", "why", "how", "when", "where", "which", "who", "whom",
        "is", "are", "was", "were", "the", "a", "an", "of", "to", "in",
        "on", "for", "and", "or", "with", "from", "by", "does", "do",
        "can", "could", "would", "should", "explain", "define", "describe",
        "difference", "between", "give", "list", "mention", "write", "about"
    }

    words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    return {word for word in words if word not in stop_words and len(word) > 2}


def _score_sentence(sentence, question_terms):
    """Score a sentence using lexical overlap with the question."""
    words = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
    if not words or not question_terms:
        return 0.0

    overlap = len(words & question_terms)
    return overlap / len(question_terms)


def extract_answer(question, retrieved_chunks, max_sentences=5):
    """
    Build an answer only from sentences that already exist in retrieved notes.

    No generative model or external knowledge is used. The returned answer is
    therefore extractive: source wording is preserved rather than invented.
    """
    question_terms = _question_terms(question)

    candidates = []

    for chunk_rank, chunk in enumerate(retrieved_chunks):
        for sentence_rank, sentence in enumerate(_split_sentences(chunk["text"])):
            score = _score_sentence(sentence, question_terms)

            # Give earlier retrieved chunks a small ranking advantage.
            score += max(0.0, 0.01 * (len(retrieved_chunks) - chunk_rank))

            if score > 0:
                candidates.append({
                    "sentence": sentence,
                    "score": score,
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "source_type": chunk.get("source_type", "unknown"),
                    "sentence_rank": sentence_rank,
                })

    if not candidates:
        return NOT_FOUND_MESSAGE

    candidates.sort(key=lambda item: item["score"], reverse=True)

    # Keep the original source order where possible, while selecting the
    # strongest evidence first and avoiding duplicate sentences.
    selected = []
    seen_sentences = set()

    for candidate in candidates:
        normalized = candidate["sentence"].lower()
        if normalized in seen_sentences:
            continue

        selected.append(candidate)
        seen_sentences.add(normalized)

        if len(selected) >= max_sentences:
            break

    if not selected:
        return NOT_FOUND_MESSAGE

    selected.sort(key=lambda item: (item["chunk_id"], item["sentence_rank"]))

    return "\n".join(f"• {item['sentence']}" for item in selected)


def generate_answer(question, retrieved_chunks):
    """Backward-compatible name for the local extractive answer generator."""
    return extract_answer(question, retrieved_chunks)
