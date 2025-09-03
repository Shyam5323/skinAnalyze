from dotenv import load_dotenv
load_dotenv()

import logging
import os
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Remove the record_audio function since Gradio handles audio recording
# Keep only the transcription function

def transcribe_with_groq(stt_model: str, audio_filepath: str) -> str:
    """Transcribe audio using Groq Whisper. Pulls key from env automatically."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your environment or .env file.")
    client = Groq(api_key=api_key)
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )
    return transcription.text

stt_model = "whisper-large-v3"