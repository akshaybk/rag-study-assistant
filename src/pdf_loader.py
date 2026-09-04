import pymupdf


def load_pdf(pdf_path):
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        list of dictionaries containing:
        - text
        - page
        - source
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number,
                "source": pdf_path
            })

    document.close()

    return pages