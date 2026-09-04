from src.retriever import retrieve
from src.generate import generate_answer


NOT_FOUND_MESSAGE = "Information not found in the provided notes."


def answer_question(question, index, chunks, top_k=3):
    """
    Retrieve relevant chunks and build an extractive answer.

    The answer generator uses only text already present in the retrieved
    source chunks. No LLM or external API is used.
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
    # 3. EXTRACT ANSWER LOCALLY
    # =========================

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    # =========================
    # 4. EXTRACT SOURCE INFO
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
    # 5. RETURN STRUCTURED DATA
    # =========================

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
