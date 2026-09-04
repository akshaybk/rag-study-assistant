from pathlib import Path

from src.pdf_loader import load_pdf
from src.chunker import chunk_pages
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store


def build_knowledge_base(pdf_paths, chunk_size=500, overlap=100):
    """Build a local FAISS knowledge base from one or more PDF files.

    Args:
        pdf_paths: Iterable of PDF paths.
        chunk_size: Maximum approximate character count per chunk.
        overlap: Character overlap between adjacent chunks on a page.

    Returns:
        A dictionary containing pages, chunks, embeddings, index, and source
        paths. All processing is local; no LLM or external API is required.
    """
    paths = [Path(path) for path in pdf_paths]
    if not paths:
        raise ValueError("At least one PDF file is required.")

    for path in paths:
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

    all_pages = []

    for path in paths:
        pages = load_pdf(str(path))
        if not pages:
            raise ValueError(f"No extractable text found in PDF: {path}")
        all_pages.extend(pages)

    chunks = chunk_pages(
        all_pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError("No text chunks could be created from the PDFs.")

    embeddings = create_embeddings(chunks)
    index = create_vector_store(embeddings)

    return {
        "sources": [str(path) for path in paths],
        "pages": all_pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
    }
