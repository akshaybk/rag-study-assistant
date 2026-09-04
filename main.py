from pathlib import Path

from src.knowledge_base import build_knowledge_base
from src.rag_pipeline import answer_question
from src.pdf_generator import (
    generate_answers_pdf,
    generate_source_mapping_pdf,
)


DATA_DIR = Path("data")


# =========================
# 1. BUILD KNOWLEDGE BASE
# =========================

pdf_paths = sorted(DATA_DIR.glob("*.pdf"))

if not pdf_paths:
    raise FileNotFoundError(
        "No PDF files found in the data folder. "
        "Add at least one source PDF to data/."
    )

knowledge_base = build_knowledge_base(pdf_paths)

pages = knowledge_base["pages"]
chunks = knowledge_base["chunks"]
embeddings = knowledge_base["embeddings"]
index = knowledge_base["index"]

for pdf_path in pdf_paths:
    page_count = sum(
        1 for page in pages if Path(page["source"]).resolve() == pdf_path.resolve()
    )
    print(f"Loaded {page_count} pages from {pdf_path}")

print(f"Total source PDFs: {len(pdf_paths)}")
print(f"Total pages loaded: {len(pages)}")
print(f"Created {len(chunks)} chunks")
print(f"Embedding shape: {embeddings.shape}")
print(f"FAISS index contains {index.ntotal} vectors")


# =========================
# 2. QUESTIONS
# =========================

questions = [
    "What is an algorithm?",
    "Why is algorithm efficiency important?",
    "What is the difference between linear search and binary search?",
    "What is quantum computing?",
]


# =========================
# 3. PROCESS QUESTIONS
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
        top_k=3,
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
# 4. GENERATE PDF REPORTS
# =========================

answers_pdf = generate_answers_pdf(all_results)
source_mapping_pdf = generate_source_mapping_pdf(all_results)

print("\n")
print("=" * 70)
print("PDF REPORTS GENERATED")
print("=" * 70)

print(f"Answers PDF: {answers_pdf}")
print(f"Source Mapping PDF: {source_mapping_pdf}")

print("\n")
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(f"Questions processed: {len(all_results)}")
