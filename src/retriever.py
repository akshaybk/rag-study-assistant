import numpy as np

from src.embeddings import model


def retrieve(query, index, chunks, top_k=3):
    """
    Find the most relevant chunks for a question.
    """

    # Convert the question into an embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, index_position in zip(distances[0], indices[0]):

        chunk = chunks[index_position]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
            "distance": float(distance)
        })

    return results