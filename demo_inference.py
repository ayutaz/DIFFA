"""
Demo inference script for DIFFA model.
Takes a single audio file and an optional question, then generates a response.

Usage:
    uv run python demo_inference.py \
        --audio_path ./sample.wav \
        --question "What is the speaker saying?" \
        --steps 16 --block_length 16 --max_new_tokens 64
"""

import argparse
import numpy as np
import torch
import soundfile as sf
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel
from loguru import logger

from src.modeling_whisper_encoder import WhisperEncoder
from src.modeling_DIFFA import DIFFAModel


def load_model(model_path, whisper_path, llm_path):
    logger.info("Loading tokenizer and feature extractor...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    logger.info("Loading DIFFA adapter weights...")
    model = DIFFAModel.from_pretrained(model_path, tokenizer=tokenizer).cuda()
    model.stage = 2

    logger.info("Loading Whisper encoder...")
    model.whisper_model = WhisperEncoder.from_pretrained(whisper_path).cuda()

    logger.info("Loading LLaDA-8B with device_map='auto' (CPU/GPU offloading)...")
    model.llm_model = AutoModel.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()
    logger.info("All models loaded successfully.")
    return model, tokenizer, feature_extractor


def read_audio(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def generate_response(model, tokenizer, feature_extractor, audio_array, instruction, args):
    audio_tensor = torch.tensor(audio_array)
    inputs = feature_extractor(
        audio_tensor.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    ).to("cuda")

    with torch.no_grad():
        response = model.generate(
            tokenizer,
            input_audio_features=inputs.input_features,
            input_audio_mask=inputs.attention_mask,
            system_prompt="You are a helpful voice assistant.",
            input_texts=[instruction],
            steps=args.steps,
            block_length=args.block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=126336,
            max_new_tokens=args.max_new_tokens,
        )
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="DIFFA demo inference on a single audio file")

    parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file (WAV, 16kHz preferred)")
    parser.add_argument("--question", type=str, default="What is the speaker saying?", help="Question/instruction about the audio")

    parser.add_argument("--model_path", type=str, default="./models/DIFFA", help="Path to DIFFA adapter checkpoint")
    parser.add_argument("--llm_path", type=str, default="./models/LLaDA-8B-Instruct", help="Path to LLaDA-8B-Instruct")
    parser.add_argument("--whisper_path", type=str, default="./models/whisper-small", help="Path to Whisper-Small")

    parser.add_argument("--steps", type=int, default=16, help="Diffusion sampling steps")
    parser.add_argument("--block_length", type=int, default=16, help="Block length for semi-autoregressive generation")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum new tokens to generate")

    args = parser.parse_args()

    model, tokenizer, feature_extractor = load_model(args.model_path, args.whisper_path, args.llm_path)

    logger.info(f"Reading audio: {args.audio_path}")
    audio_array = read_audio(args.audio_path)
    duration = len(audio_array) / 16000
    logger.info(f"Audio duration: {duration:.2f}s")

    logger.info(f"Question: {args.question}")
    output = generate_response(model, tokenizer, feature_extractor, audio_array, args.question, args)

    print(f"\n{'='*60}")
    print(f"Audio: {args.audio_path}")
    print(f"Question: {args.question}")
    print(f"Response: {output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
