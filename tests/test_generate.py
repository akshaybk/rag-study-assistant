import numpy as np

from src import generate


class FakeEmbeddingModel:
    """Small deterministic stand-in so unit tests do not need model inference."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array([
                float("algorithm" in lowered),
                float("finite" in lowered),
                float("binary" in lowered),
                float("linear" in lowered),
                float("structure" in lowered),
                float("prime" in lowered),
                float("dictionary" in lowered),
                float("unrelated" in lowered),
            ], dtype="float32")
            norm = np.linalg.norm(vector)
            if normalize_embeddings and norm:
                vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors)


def test_unsupported_question_fails_closed():
    answer, evidence = generate.extract_answer_with_evidence(
        "What is quantum computing?",
        [],
    )

    assert answer == generate.NOT_FOUND_MESSAGE
    assert evidence == []


def test_answer_uses_source_sentence_and_returns_exact_evidence(monkeypatch):
    monkeypatch.setattr(generate, "model", FakeEmbeddingModel())

    chunks = [
        {
            "chunk_id": 7,
            "page": 2,
            "source": "notes.pdf",
            "source_type": "study_notes",
            "text": (
                "An algorithm is a finite sequence of clear instructions. "
                "This unrelated sentence should not be selected."
            ),
        }
    ]

    answer, evidence = generate.extract_answer_with_evidence(
        "What is an algorithm?",
        chunks,
        max_sentences=2,
    )

    assert "An algorithm is a finite sequence of clear instructions." in answer
    assert "This unrelated sentence should not be selected." not in answer
    assert len(evidence) == 1
    assert evidence[0]["sentence"] == (
        "An algorithm is a finite sequence of clear instructions."
    )
    assert evidence[0]["page"] == 2
    assert evidence[0]["chunk_id"] == 7


def test_question_paper_heading_is_not_returned_as_answer(monkeypatch):
    monkeypatch.setattr(generate, "model", FakeEmbeddingModel())

    chunks = [
        {
            "chunk_id": 33,
            "page": 12,
            "source": "notes.pdf",
            "source_type": "study_notes",
            "text": (
                "Q2 — What is a structure in C++? "
                "A structure (struct) is a user-defined data type."
            ),
        },
        {
            "chunk_id": 38,
            "page": 13,
            "source": "notes.pdf",
            "source_type": "study_notes",
            "text": (
                "Q9 — Structures vs Classes in C++ Structure Class "
                "Declared using struct Declared using class."
            ),
        },
    ]

    answer, evidence = generate.extract_answer_with_evidence(
        "What is a structure?",
        chunks,
        max_sentences=3,
    )

    assert "Q2 — What is a structure in C++?" not in answer
    assert "Q9 — Structures vs Classes in C++" not in answer
    assert "A structure (struct) is a user-defined data type." in answer
    assert all(not item["sentence"].startswith("Q") for item in evidence)


def test_previous_question_paper_is_not_used_as_evidence(monkeypatch):
    monkeypatch.setattr(generate, "model", FakeEmbeddingModel())

    chunks = [
        {
            "chunk_id": 1,
            "page": 4,
            "source": "old-paper.pdf",
            "source_type": "previous_question_paper",
            "text": "An algorithm is a finite sequence of clear instructions.",
        },
        {
            "chunk_id": 2,
            "page": 8,
            "source": "notes.pdf",
            "source_type": "study_notes",
            "text": "An algorithm is a step-by-step procedure for solving a problem.",
        },
    ]

    answer, evidence = generate.extract_answer_with_evidence(
        "What is an algorithm?",
        chunks,
        max_sentences=2,
    )

    assert "old-paper.pdf" not in answer
    assert all(item["source_type"] == "study_notes" for item in evidence)
    assert "step-by-step procedure" in answer


def test_multi_part_question_is_answered_independently(monkeypatch):
    monkeypatch.setattr(generate, "model", FakeEmbeddingModel())

    chunks = [
        {
            "chunk_id": 1,
            "page": 10,
            "source": "notes.pdf",
            "source_type": "study_notes",
            "text": "A prime number is divisible only by one and itself.",
        },
        {
            "chunk_id": 2,
            "page": 14,
            "source": "notes.pdf",
            "source_type": "study_notes",
            "text": "A dictionary stores key-value pairs for lookup.",
        },
    ]

    question = "a) Define a prime number. (6) b) Explain a dictionary. (6)"
    answer, evidence = generate.extract_answer_with_evidence(
        question,
        chunks,
        max_sentences=2,
    )

    assert "a)" in answer
    assert "b)" in answer
    assert "prime number" in answer
    assert "dictionary" in answer
    assert len(evidence) == 2


def test_split_subquestions():
    question = "a) What is a prime number? (6) b) Explain a dictionary. (6)"
    assert generate.split_subquestions(question) == [
        ("a", "What is a prime number? (6)"),
        ("b", "Explain a dictionary. (6)"),
    ]


def test_comparison_prefers_sentence_containing_both_terms(monkeypatch):
    monkeypatch.setattr(generate, "model", FakeEmbeddingModel())

    chunks = [
        {
            "chunk_id": 1,
            "page": 3,
            "source": "notes.pdf",
            "source_type": "study_notes",
            "text": (
                "Linear search checks elements sequentially. "
                "Binary search divides the search range into halves. "
                "Linear search and binary search have different time complexities."
            ),
        }
    ]

    answer, evidence = generate.extract_answer_with_evidence(
        "What is the difference between linear search and binary search?",
        chunks,
        max_sentences=1,
    )

    assert "Linear search and binary search" in answer
    assert len(evidence) == 1
