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
        "GEMINI_API_KEY not found. "
        "Check your .env file."
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
# GENERATE ANSWER
# =========================

def generate_answer(question, context):

    prompt = f"""
You are an academic study assistant.

Answer the user's question using ONLY the
SOURCE MATERIAL provided below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the source material does not contain enough
   information, say:

   "Information not found in the provided notes."

4. Write a clear, exam-ready answer.
5. Use headings and bullet points where appropriate.
6. Keep the answer faithful to the source material.

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

    # Try available models
    for model in MODELS:

        print(f"\nTrying model: {model}")

        # Retry temporary server failures
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

                print(
                    f"Attempt {attempt + 1}/3 failed: {error}"
                )

                # Wait before retrying
                if attempt < 2:
                    time.sleep(3)

        print(f"Model {model} failed. Trying next model...")

    # All models failed
    raise RuntimeError(
        f"All Gemini models failed.\n"
        f"Last error: {last_error}"
    )