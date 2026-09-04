from pathlib import Path
import tempfile

import streamlit as st

from src.knowledge_base import build_knowledge_base
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
    "Upload your study-note PDFs, enter questions, and generate exam-ready "
    "answers with a source mapping report. Answers are extracted only from "
    "the uploaded notes."
)


uploaded_pdfs = st.file_uploader(
    "Upload study-note PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload one or multiple PDFs.",
)

question_text = st.text_area(
    "Enter your questions",
    placeholder=(
        "Enter one question per line, for example:\n"
        "What is an algorithm?\n"
        "Why is algorithm efficiency important?"
    ),
    height=160,
)


if st.button("Generate Answers", type="primary", disabled=not uploaded_pdfs):
    questions = [line.strip() for line in question_text.splitlines() if line.strip()]

    if not questions:
        st.warning("Please enter at least one question.")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_paths = []

        for uploaded_file in uploaded_pdfs:
            destination = temp_path / uploaded_file.name
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
                result = answer_question(
                    question,
                    index,
                    chunks,
                    top_k=3,
                )
                results.append(result)
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
