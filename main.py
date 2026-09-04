from src.pdf_loader import load_pdf
from src.chunker import chunk_pages
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.rag_pipeline import answer_question


PDF_PATH = "data/notes.pdf"


# =========================
# 1. LOAD PDF
# =========================

pages = load_pdf(PDF_PATH)

print(f"Loaded {len(pages)} pages")


# =========================
# 2. CREATE CHUNKS
# =========================

chunks = chunk_pages(pages)

print(f"Created {len(chunks)} chunks")


# =========================
# 3. CREATE EMBEDDINGS
# =========================

embeddings = create_embeddings(chunks)

print(f"Embedding shape: {embeddings.shape}")


# =========================
# 4. CREATE VECTOR STORE
# =========================

index = create_vector_store(embeddings)

print(f"FAISS index contains {index.ntotal} vectors")


# =========================
# 5. QUESTIONS
# =========================

questions = [
    "What is an algorithm?",
    "Why is algorithm efficiency important?",
    "What is the difference between linear search and binary search?"
]


# =========================
# 6. PROCESS QUESTIONS
# =========================

all_results = []


for number, question in enumerate(questions, start=1):

    print("\n")
    print("=" * 70)
    print(f"QUESTION {number}")
    print(question)
    print("=" * 70)

    result = answer_question(
        question,
        index,
        chunks,
        top_k=3
    )

    all_results.append(result)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:

        print(
            f"  {source['file']} "
            f"→ Page {source['page']} "
            f"(distance: {source['distance']:.4f})"
        )


# =========================
# 7. SUMMARY
# =========================

print("\n")
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(f"Questions processed: {len(all_results)}")