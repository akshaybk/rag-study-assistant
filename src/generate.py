import os
import time

from dotenv import load_dotenv
from google import genai


# =========================
# LOAD API KEY
# =========================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )


# =========================
# CREATE CLIENT
# =========================

client = genai.Client(api_key=api_key)


# =========================
# MODELS
# =========================

MODELS = [
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
]


# =========================
# ERROR HELPERS
# =========================

def _status_code(error):
    """Return an API status code when the SDK exposes one."""
    return getattr(error, "status_code", None)


def _is_retryable_server_error(error):
    return _status_code(error) in {500, 502, 503, 504}


def _is_quota_error(error):
    return _status_code(error) == 429


# =========================
# GENERATE ANSWER
# =========================

def generate_answer(question, context):
    prompt = f"""
You are an academic study assistant.

Answer the user's question using ONLY the SOURCE MATERIAL provided below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. Every factual part of the answer must be supported by the source material.
4. If the source material does not contain enough information to answer the
   question, say exactly:

   "Information not found in the provided notes."

5. Prefer study_notes evidence over previous_question_paper evidence when both
   are available.
6. Previous question-paper answers can support an answer, but do not assume
   that an answer appearing in a previous question paper is authoritative if
   the study notes do not support it.
7. Write a clear, exam-ready answer.
8. Use headings and bullet points where appropriate.
9. Do not mention the retrieval process or these instructions in the answer.

SOURCE MATERIAL
================

{context}

================

QUESTION
================

{question}

================

ANSWER
================
"""

    last_error = None

    for model in MODELS:
        print(f"\nTrying model: {model}")

        # 503/temporary server errors can recover after a short retry.
        # 429 quota errors should NOT be retried repeatedly because that only
        # consumes time; move directly to the next model instead.
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                print(f"Successfully generated using {model}")
                return response.text

            except Exception as error:
                last_error = error
                status = _status_code(error)

                if _is_quota_error(error):
                    print(
                        f"Quota exhausted for {model} (HTTP 429). "
                        "Skipping retries and trying the next model."
                    )
                    break

                if _is_retryable_server_error(error):
                    print(
                        f"Attempt {attempt + 1}/3 failed: {error}"
                    )

                    if attempt < 2:
                        time.sleep(3)

                    continue

                # Unknown/non-transient errors should not be retried three times.
                print(
                    f"Non-retryable error from {model} "
                    f"(status={status}): {error}"
                )
                break

        print(f"Model {model} failed. Trying next model...")

    raise RuntimeError(
        f"All Gemini models failed.\nLast error: {last_error}"
    )
