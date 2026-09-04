import faiss
import numpy as np


def create_vector_store(embeddings):
    """
    Create a FAISS vector index from embeddings.
    """

    # Make sure embeddings are float32
    embeddings = np.array(embeddings).astype("float32")

    # Number of dimensions in each embedding
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(embeddings)

    return index