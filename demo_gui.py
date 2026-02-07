"""
Gradio Web UI for DIFFA model demo.
Upload an audio file and ask a question to get a response.

Usage:
    uv run python demo_gui.py
    # Then open http://localhost:7860 in your browser
"""

from types import SimpleNamespace

import gradio as gr
from loguru import logger

from demo_inference import load_model, read_audio, generate_response

MODEL_PATH = "./models/DIFFA"
WHISPER_PATH = "./models/whisper-small"
LLM_PATH = "./models/LLaDA-8B-Instruct"

DEFAULT_ARGS = SimpleNamespace(steps=16, block_length=16, max_new_tokens=64)

logger.info("Loading models (this may take a few minutes)...")
model, tokenizer, feature_extractor = load_model(MODEL_PATH, WHISPER_PATH, LLM_PATH)
logger.info("Models loaded. Starting Gradio server...")


def predict(audio_path, question):
    if audio_path is None:
        return "Please upload an audio file."
    if not question or not question.strip():
        question = "What is the speaker saying?"

    audio_array = read_audio(audio_path)
    duration = len(audio_array) / 16000
    logger.info(f"Audio duration: {duration:.2f}s | Question: {question}")

    response = generate_response(
        model, tokenizer, feature_extractor, audio_array, question, DEFAULT_ARGS
    )
    return response


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Audio(type="filepath", label="Audio"),
        gr.Textbox(
            label="Question",
            placeholder="What is the speaker saying?",
            value="What is the speaker saying?",
        ),
    ],
    outputs=gr.Textbox(label="Response"),
    title="DIFFA Demo",
    description="Upload an audio file and ask a question. The model will listen and respond.",
)

if __name__ == "__main__":
    demo.launch()
