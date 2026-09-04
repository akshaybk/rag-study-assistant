from src.retriever import retrieve
from src.generate import generate_answer


NOT_FOUND_MESSAGE = "Information not found in the provided notes."


def answer_question(question, index, chunks, top_k=3):
    """
    Retrieve relevant chunks and generate an answer for a single question.

    If no sufficiently relevant source evidence is found, the LLM is not
    called and the required not-found message is returned.
    """

    # =========================
    # 1. RETRIEVE
    # =========================

    retrieved_chunks = retrieve(
        question,
        index,
        chunks,
        top_k=top_k
    )

    # =========================
    # 2. HANDLE NO EVIDENCE
    # =========================

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": NOT_FOUND_MESSAGE,
            "sources": []
        }

    # =========================
    # 3. BUILD CONTEXT
    # =========================

    context_parts = []

    for result in retrieved_chunks:
        context_parts.append(
            f"""
SOURCE: {result['source']}
SOURCE TYPE: {result['source_type']}
PAGE: {result['page']}
RELEVANCE: {result['relevance']:.4f}

{result['text']}
"""
        )

    context = "\n\n".join(context_parts)

    # =========================
    # 4. GENERATE ANSWER
    # =========================

    answer = generate_answer(
        question,
        context
    )

    # =========================
    # 5. EXTRACT SOURCE INFO
    # =========================

    sources = []

    for result in retrieved_chunks:
        sources.append({
            "file": result["source"],
            "page": result["page"],
            "chunk_id": result["chunk_id"],
            "source_type": result["source_type"],
            "distance": result["distance"],
            "relevance": result["relevance"],
            "text": result["text"]
        })

    # =========================
    # 6. RETURN STRUCTURED DATA
    # =========================

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
