import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs

SYSTEM_PROMPT = (
    "You have to act as a professional doctor (educational setting). "
    "What's in this image? Do you find anything wrong medically? "
    "If you make a differential, suggest some remedies. No numbers, "
    "one concise paragraph (max 2 sentences). Start directly."
)

def process_inputs(audio_filepath, image_filepath):
    stt_text = transcribe_with_groq(
        stt_model="whisper-large-v3",
        audio_filepath=audio_filepath
    )
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=SYSTEM_PROMPT + " " + stt_text,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            encoded_image=encode_image(image_filepath)
        )
    else:
        doctor_response = "No image provided."
    voice_path = text_to_speech_with_elevenlabs(
        input_text=doctor_response,
        output_filepath="final.mp3"
    )
    return stt_text, doctor_response, voice_path

demo = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Patient Audio"),
        gr.Image(type="filepath", label="Photo of Concern")
    ],
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Doctor Response"),
        gr.Audio(label="Doctor Voice")
    ],
    title="AI Doctor (Vision + Voice Demo)"
)

if __name__ == "__main__":
    demo.launch()