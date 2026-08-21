from src.config import settings
from google import genai
import sys

client = genai.Client(api_key=settings.gemini_api_key)
try:
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents='Hello'
    )
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
