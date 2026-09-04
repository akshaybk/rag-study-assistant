def chunk_pages(pages, chunk_size=500, overlap=100):
    """
    Split extracted PDF pages into smaller overlapping chunks.

    Each chunk keeps its source and page number.
    """

    chunks = []

    chunk_id = 0

    for page in pages:

        text = page["text"]

        # Clean unnecessary whitespace
        text = " ".join(text.split())

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"]
            })

            chunk_id += 1

            # Move forward while keeping some overlap
            start += chunk_size - overlap

    return chunks