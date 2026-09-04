import numpy as np

from src.embeddings import model


# FAISS uses L2 distance. Smaller values mean greater similarity.
# This is intentionally configurable so we can tune it using real notes.
DEFAULT_MAX_DISTANCE = 1.15


def retrieve(query, index, chunks, top_k=3, max_distance=DEFAULT_MAX_DISTANCE):
    """
    Find relevant chunks for a question.

    Chunks whose FAISS distance is above max_distance are discarded.
    This prevents weakly related chunks from being passed to the LLM.

    Returns retrieved chunks with an intuitive relevance score as well as
    the original FAISS distance.
    """

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    seen = set()

    for distance, index_position in zip(distances[0], indices[0]):
        if index_position < 0:
            continue

        distance = float(distance)

        if distance > max_distance:
            continue

        chunk = chunks[index_position]

        # Avoid returning the same file/page repeatedly when overlapping
        # chunks point to essentially the same source location.
        source_key = (chunk["source"], chunk["page"])
        if source_key in seen:
            continue

        seen.add(source_key)

        # Convert distance into a simple 0-1 relevance score.
        relevance = 1.0 / (1.0 + distance)

        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
            "source_type": chunk.get("source_type", "unknown"),
            "distance": distance,
            "relevance": relevance,
        })

    return results
