from src.pdf_loader import load_pdf


PDF_PATH = "data/notes.pdf"


pages = load_pdf(PDF_PATH)

print(f"Loaded {len(pages)} pages\n")

for page in pages[:3]:

    print("=" * 60)
    print(f"PAGE: {page['page']}")
    print("=" * 60)

    print(page["text"][:1000])