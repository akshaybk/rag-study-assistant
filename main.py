from pathlib import Path

from src.pdf_loader import load_pdf
from src.chunker import chunk_pages
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.rag_pipeline import answer_question
from src.pdf_generator import (
    generate_answers_pdf,
    generate_source_mapping_pdf,
)


DATA_DIR = Path("data")


# =========================
# 1. LOAD SOURCE PDFs
# =========================

pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

if not pdf_paths:
    raise FileNotFoundError(
        "No PDF files found in the data folder. "
        "Add at least one source PDF to data/."
    )


all_pages = []

for pdf_path in pdf_paths:
    pages = load_pdf(str(pdf_path))
    all_pages.extend(pages)
    print(f"Loaded {len(pages)} pages from {pdf_path}")

print(f"Total source PDFs: {len(pdf_paths)}")
print(f"Total pages loaded: {len(all_pages)}")


# =========================
# 2. CREATE CHUNKS
# =========================

chunks = chunk_pages(all_pages)

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
    "What is the difference between linear search and binary search?",
    "What is quantum computing?"
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
# 7. GENERATE PDF REPORTS
# =========================

answers_pdf = generate_answers_pdf(all_results)
source_mapping_pdf = generate_source_mapping_pdf(all_results)

print("\n")
print("=" * 70)
print("PDF REPORTS GENERATED")
print("=" * 70)

print(f"Answers PDF: {answers_pdf}")
print(f"Source Mapping PDF: {source_mapping_pdf}")


# =========================
# 8. SUMMARY
# =========================

print("\n")
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(f"Questions processed: {len(all_results)}")
