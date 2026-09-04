from src.retriever import retrieve
from src.generate import generate_answer


def answer_question(question, index, chunks, top_k=3):
    """
    Retrieve relevant chunks and generate an answer
    for a single question.
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
    # 2. BUILD CONTEXT
    # =========================

    context_parts = []

    for result in retrieved_chunks:

        context_parts.append(
            f"""
SOURCE: {result['source']}
PAGE: {result['page']}

{result['text']}
"""
        )

    context = "\n\n".join(context_parts)

    # =========================
    # 3. GENERATE ANSWER
    # =========================

    answer = generate_answer(
        question,
        context
    )

    # =========================
    # 4. EXTRACT SOURCE INFO
    # =========================

    sources = []

    for result in retrieved_chunks:

        source = {
            "file": result["source"],
            "page": result["page"],
            "chunk_id": result["chunk_id"],
            "distance": result["distance"]
        }

        sources.append(source)

    # =========================
    # 5. RETURN STRUCTURED DATA
    # =========================

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }