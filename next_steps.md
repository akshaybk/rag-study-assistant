# Next Steps

This document tracks the next development milestones for the RAG Study Assistant.

## 1. Improve the Answer Quality Engine

Focus on making retrieved content produce clean, relevant, exam-ready answers instead of simply returning loosely related sentences.

- Improve semantic + lexical relevance scoring.
- Improve topic matching between the question and retrieved chunks.
- Select only evidence that directly answers the question.
- Remove unrelated sentences from the final answer.
- Handle different question types properly:
  - Definitions
  - Explanations
  - Why / importance questions
  - Characteristics
  - Advantages / disadvantages
  - Applications
  - Steps / procedures
  - Comparisons
  - Examples
  - Code-related questions
- Preserve the terminology and explanations used in the provided notes.
- Keep the answers concise and exam-ready.

## 2. Robust Question Parsing

Make question extraction reliable for both typed questions and OCR output from question-paper images.

- Handle numbered questions such as `1.`, `Q1`, `Q1.` etc.
- Handle subquestions such as `(a)`, `(b)`, `a)`, `b)` etc.
- Remove mark allocations such as `(6)`, `[6]`, `6 marks` from the question text.
- Handle multi-line questions correctly.
- Preserve the original question structure where useful.
- Clean common OCR errors without changing the meaning of the question.
- Ensure multipart questions are answered independently.

## 3. Hybrid Retrieval

Improve retrieval by combining semantic similarity with keyword/topic matching.

- Keep FAISS vector search as the semantic retrieval layer.
- Add stronger lexical matching for important terms.
- Give preference to chunks containing the key concepts in the question.
- Retrieve a wider candidate set before selecting final evidence.
- Deduplicate overlapping chunks.
- Keep page and source metadata attached to every retrieved chunk.

## 4. Confidence and Evidence Validation

Strengthen the anti-hallucination behaviour of the system.

- Validate that the selected evidence actually supports the generated answer.
- Calculate a confidence/relevance score for each answer.
- Reject weak or unrelated evidence.
- Return exactly:

> `Information not found in the provided notes.`

when the provided notes do not contain enough supporting information.

- Make sure unsupported topics such as unrelated syllabus units are not answered using general model knowledge.

## 5. Improve Source Mapping

Make the source mapping PDF more useful for studying and verification.

For every question, clearly show:

- Question
- Answer status
- Source PDF / file
- Source page number
- Retrieved chunk
- Supporting evidence
- Relevance / confidence score

The mapping should make it easy to verify where every answer came from.

## 6. Knowledge Base Caching

Avoid rebuilding the entire knowledge base every time the application starts.

- Detect whether the source PDFs have changed.
- Cache processed chunks and embeddings.
- Reuse existing FAISS indexes when the source documents are unchanged.
- Rebuild only when files are added, removed, or modified.
- Keep the cache local so the project remains fully offline.

## 7. Final Streamlit UI

Once the answer engine is reliable, polish the user-facing application.

- Upload one or multiple source-note PDFs.
- Enter questions manually.
- Upload a question-paper image.
- Preview and edit OCR-extracted questions.
- Generate answers.
- Display evidence and source information.
- Download the Answers PDF.
- Download the Source Mapping PDF.
- Show clear errors and unsupported-question messages.

## 8. Testing and Quality Checks

Expand automated tests as each milestone is completed.

- Unit tests for question parsing.
- Unit tests for answer extraction.
- Tests for unsupported questions.
- Tests for multipart questions.
- Tests for retrieval ranking.
- Tests for source/page metadata preservation.
- Tests for PDF generation.
- Run the complete test suite before major milestones.

## Recommended Order

The implementation order should be:

1. **Answer Quality Engine**
2. **Robust Question Parsing**
3. **Hybrid Retrieval**
4. **Confidence / Evidence Validation**
5. **Source Mapping Improvements**
6. **Knowledge Base Caching**
7. **Final Streamlit UI Polish**
8. **Testing and Final Cleanup**

The immediate next task is **Step 1: Improve the Answer Quality Engine**. The PDF presentation layer has already been cleaned up; the next priority is improving the actual relevance and organization of the answers.