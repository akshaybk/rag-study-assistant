from src.pdf_loader import load_pdf
from src.chunker import chunk_pages
from src.embeddings import create_embeddings


PDF_PATH = "data/notes.pdf"


# -------------------------
# STEP 1: Load PDF
# -------------------------

pages = load_pdf(PDF_PATH)

print(f"Loaded {len(pages)} pages")


# -------------------------
# STEP 2: Create chunks
# -------------------------

chunks = chunk_pages(pages)

print(f"Created {len(chunks)} chunks")

embeddings = create_embeddings(chunks)

print(f"Embedding shape: {embeddings.shape}")

# -------------------------
# Display some chunks
# -------------------------

for chunk in chunks[:5]:

    print("\n" + "=" * 70)

    print(f"CHUNK ID : {chunk['chunk_id']}")
    print(f"SOURCE   : {chunk['source']}")
    print(f"PAGE     : {chunk['page']}")

    print("-" * 70)

    print(chunk["text"])