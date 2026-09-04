import os

from dotenv import load_dotenv
from google import genai


# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")


# Create Gemini client
client = genai.Client(api_key=api_key)


# Simple test
response = client.models.generate_content(
    model="gemini-3.7-flash",
    contents="Say hello in one sentence."
)

print("\nGemini response:")
print(response.text)