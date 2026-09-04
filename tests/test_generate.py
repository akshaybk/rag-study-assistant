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
