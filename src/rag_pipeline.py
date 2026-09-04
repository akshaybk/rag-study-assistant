from src.retriever import retrieve
from src.generate import extract_answer_with_evidence


NOT_FOUND_MESSAGE = "Information not found in the provided notes."


def answer_question(question, index, chunks, top_k=3):
    """Retrieve evidence and build a fully local extractive answer."""

    retrieved_chunks = retrieve(
        question,
        index,
        chunks,
        top_k=top_k
    )

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": NOT_FOUND_MESSAGE,
            "sources": []
        }

    answer, selected_evidence = extract_answer_with_evidence(
        question,
        retrieved_chunks
    )

    # Source mapping contains the exact sentences that were used to form the
    # answer, plus the retrieval metadata for auditability.
    selected_keys = {
        (
            item["source"],
            item["page"],
            item["chunk_id"]
        )
        for item in selected_evidence
    }

    sources = []
    added = set()

    for item in selected_evidence:
        key = (
            item["source"],
            item["page"],
            item["chunk_id"],
            item["sentence"]
        )
        if key in added:
            continue
        added.add(key)

        # Find the retrieval metadata for this evidence sentence.
        matching_chunk = next(
            (
                result for result in retrieved_chunks
                if result["source"] == item["source"]
                and result["page"] == item["page"]
                and result["chunk_id"] == item["chunk_id"]
            ),
            None
        )

        sources.append({
            "file": item["source"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "source_type": item["source_type"],
            "distance": matching_chunk["distance"] if matching_chunk else None,
            "relevance": matching_chunk["relevance"] if matching_chunk else None,
            "text": item["sentence"],
        })

    # This is normally unreachable when an answer exists, but keeps the
    # fail-closed behavior explicit.
    if not selected_evidence:
        answer = NOT_FOUND_MESSAGE

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_sources": [
            {
                "file": result["source"],
                "page": result["page"],
                "chunk_id": result["chunk_id"],
                "source_type": result["source_type"],
                "distance": result["distance"],
                "relevance": result["relevance"],
            }
            for result in retrieved_chunks
            if (
                result["source"],
                result["page"],
                result["chunk_id"]
            ) not in selected_keys
        ]
    }
