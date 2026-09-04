from src.question_parser import extract_questions


def test_extract_numbered_questions():
    text = (
        "Q1. What is an algorithm?\n"
        "Q2 — Explain binary search.\n"
        "3. What is a structure?"
    )

    questions = extract_questions(text)

    assert [item["number"] for item in questions] == [1, 2, 3]
    assert questions[0]["question"] == "What is an algorithm?"
    assert questions[1]["question"] == "Explain binary search."
    assert questions[2]["question"] == "What is a structure?"


def test_fallback_extracts_unlabeled_question_lines():
    text = "What is an algorithm?\n\nExplain binary search in detail."

    questions = extract_questions(text)

    assert len(questions) == 2
    assert questions[0]["question"] == "What is an algorithm?"
    assert questions[1]["question"] == "Explain binary search in detail."
