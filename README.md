# Unified Podcast Generator

A production-ready AI podcast generator using **Indic-ParlerTTS** (21 Indian languages, 69 voices) and self-hosted **Llama/Mistral** LLM.

## Features

- 🎙️ **21 Indian Languages**: Hindi, Telugu, Tamil, Malayalam, Kannada, Bengali, Marathi, Gujarati, and more
- 🗣️ **69 Unique Voices**: Rohit, Divya (Hindi), Prakash, Lalitha (Telugu), etc.
- 😊 **12 Emotions**: Happy, Sad, Anger, Fear, Surprise, Neutral, etc.
- 🤖 **Self-Hosted LLM**: Llama/Mistral for script generation
- 🎵 **Audio Processing**: Background music, crossfade, noise addition
- 🖥️ **Gradio UI + CLI**

## First-Time Setup (Windows)

```powershell
cd c:\pod2\unified-podcast-generator

# 1. Set HuggingFace token (one-time for model download)
$env:HF_TOKEN = "your_hf_token_here"

# 2. Download the model
huggingface-cli download ai4bharat/indic-parler-tts

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage (After Setup)

```powershell
# Run Gradio UI
python src/main.py --ui

# CLI usage
python src/main.py --text "नमस्ते" --speaker Rohit --output output.wav
python src/main.py --list-languages
python src/main.py --list-speakers hindi
```

## Model Info

| Property | Value |
|----------|-------|
| Model | [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) |
| Languages | 21 |
| Voices | 69 |
| License | Apache 2.0 |

## Credits

- **Indic-ParlerTTS**: AI4Bharat (IIT Madras) + HuggingFace
- **Audio Processing**: Ported from kokoro-podcast-generator
- **Multi-language Prompts**: Ported from Voice-Clone-Podcast
