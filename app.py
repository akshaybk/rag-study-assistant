from pathlib import Path
import tempfile

import streamlit as st

from src.knowledge_base import build_knowledge_base
from src.ocr import extract_text_from_image
from src.question_parser import extract_questions
from src.rag_pipeline import answer_question
from src.pdf_generator import (
    generate_answers_pdf,
    generate_source_mapping_pdf,
)


st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Study Assistant")
st.caption("Local, source-grounded study assistant — no LLM or API required.")

st.markdown(
    "Upload your study-note PDFs, enter questions or upload a question-paper "
    "image, and generate exam-ready answers with source mapping."
)

uploaded_pdfs = st.file_uploader(
    "Upload study-note PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload one or multiple PDFs.",
)

st.subheader("Questions")
input_mode = st.radio(
    "Choose how to provide questions",
    ["Type / paste questions", "Upload question-paper image"],
    horizontal=True,
)

question_text = ""

if input_mode == "Type / paste questions":
    question_text = st.text_area(
        "Enter your questions",
        placeholder=(
            "Enter one question per line, for example:\n"
            "What is an algorithm?\n"
            "Why is algorithm efficiency important?"
        ),
        height=160,
    )
else:
    uploaded_images = st.file_uploader(
        "Upload question-paper image(s)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="OCR runs locally using Tesseract. No image is sent to an API.",
    )

    if uploaded_images:
        ocr_text_parts = []
        try:
            for image in uploaded_images:
                extracted = extract_text_from_image(image.getvalue())
                if extracted:
                    ocr_text_parts.append(extracted)
        except Exception as exc:
            st.error(
                "Local OCR could not run. Make sure Tesseract OCR is installed "
                f"and available on PATH. Details: {exc}"
            )
            st.stop()

        raw_ocr_text = "\n\n".join(ocr_text_parts)
        parsed_questions = extract_questions(raw_ocr_text)

        if raw_ocr_text:
            with st.expander("OCR text", expanded=False):
                st.text(raw_ocr_text)

        if parsed_questions:
            question_text = "\n".join(
                item["question"] for item in parsed_questions
            )
            st.success(f"Detected {len(parsed_questions)} question(s).")
            question_text = st.text_area(
                "Review or edit detected questions",
                value=question_text,
                height=180,
            )
        elif uploaded_images:
            st.warning("No questions could be detected from the uploaded image(s).")


if st.button(
    "Generate Answers",
    type="primary",
    disabled=not uploaded_pdfs,
):
    questions = [line.strip() for line in question_text.splitlines() if line.strip()]

    if not questions:
        st.warning("Please provide at least one question.")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_paths = []

        for index, uploaded_file in enumerate(uploaded_pdfs):
            safe_name = Path(uploaded_file.name).name
            destination = temp_path / f"{index}_{safe_name}"
            destination.write_bytes(uploaded_file.getvalue())
            pdf_paths.append(destination)

        with st.status("Building local knowledge base...", expanded=True) as status:
            knowledge_base = build_knowledge_base(pdf_paths)
            status.update(label="Knowledge base ready", state="complete")

        chunks = knowledge_base["chunks"]
        index = knowledge_base["index"]

        st.info(
            f"Processed {len(uploaded_pdfs)} PDF(s), "
            f"{len(knowledge_base['pages'])} pages, "
            f"and {len(chunks)} chunks."
        )

        results = []

        with st.status("Answering questions...", expanded=True) as status:
            for question in questions:
                results.append(
                    answer_question(
                        question,
                        index,
                        chunks,
                        top_k=6,
                    )
                )
            status.update(label="Answers generated", state="complete")

        st.subheader("Answers")

        for number, result in enumerate(results, start=1):
            st.markdown(f"### Question {number}")
            st.write(result["question"])
            st.markdown("**Answer**")
            st.markdown(result["answer"])

            if result["sources"]:
                source_summary = []
                for source in result["sources"]:
                    filename = Path(str(source["file"])).name
                    source_summary.append(
                        f"{filename} — Page {source['page']} — Chunk {source['chunk_id']}"
                    )
                st.caption("Sources: " + "; ".join(source_summary))

                with st.expander("Exact evidence"):
                    for source in result["sources"]:
                        filename = Path(str(source["file"])).name
                        st.markdown(
                            f"**{filename} — Page {source['page']} — "
                            f"Chunk {source['chunk_id']}**"
                        )
                        st.write(source["text"])
            else:
                st.caption("No source evidence retrieved.")

            st.divider()

        answers_path = temp_path / "answers.pdf"
        mapping_path = temp_path / "source_mapping.pdf"

        generate_answers_pdf(results, str(answers_path))
        generate_source_mapping_pdf(results, str(mapping_path))

        st.subheader("Download Reports")

        st.download_button(
            "📄 Download Answers PDF",
            data=answers_path.read_bytes(),
            file_name="answers.pdf",
            mime="application/pdf",
        )

        st.download_button(
            "📑 Download Source Mapping PDF",
            data=mapping_path.read_bytes(),
            file_name="source_mapping.pdf",
            mime="application/pdf",
        )
