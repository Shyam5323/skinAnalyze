"""Core multimodal brain module for the AI Doctor.

Loads the GROQ API key from the environment (via python-dotenv if a .env file is present)
and exposes helper functions to encode an image and query the Groq multimodal model.
"""

from dotenv import load_dotenv
import os
import base64
from groq import Groq
from typing import List, Dict, Any, cast

# Load .env only once (safe to call multiple times, it will noop after first load)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    # Fail fast with a clear message instead of a GroqError later.
    raise RuntimeError("GROQ_API_KEY not set. Add it to your environment or .env file.")


def encode_image(image_path: str) -> str:
    """Return base64 encoded string for the image at image_path."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_query(query: str, model: str, encoded_image: str) -> str:
    """Send a multimodal (text+image) prompt to the Groq API and return the text response."""
    client = Groq(api_key=GROQ_API_KEY)
    # Construct messages in the structure the Groq client expects.
    # Groq expects each message to conform to ChatCompletionUserMessageParam etc.
    # We'll build it as a dict and cast to the expected type to silence type check noise.
    user_message: Dict[str, Any] = {
        "role": "user",
        "content": [
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
        ],
    }
    messages = cast(List[Any], [user_message])
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    content = chat_completion.choices[0].message.content
    if content is None:
        return "(No content returned)"
    return content

# Provide a module-level default model (can be overridden by caller)
DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

