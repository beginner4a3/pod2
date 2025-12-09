# Unified Podcast Generator

A production-ready AI podcast generator using **Indic-ParlerTTS** (21 Indian languages, 69 voices) with **self-hosted LLM** for script generation and **optional voice cloning**.

## Features

- 🎙️ **21 Indian Languages**: Hindi, Telugu, Tamil, Malayalam, Kannada, Bengali, Marathi, Gujarati, and more
- 🗣️ **69 Unique Voices**: Rohit, Divya (Hindi), Prakash, Lalitha (Telugu), etc.
- 😊 **12 Emotions**: Happy, Sad, Anger, Fear, Surprise, Neutral, etc.
- 🤖 **Self-Hosted LLM**: Llama/Mistral for script generation (no API keys needed)
- 📄 **Document Upload**: PDF, DOCX, TXT support
- 🎤 **Voice Cloning**: Optional XTTS v2 for custom voices
- 🎛️ **Advanced Audio Settings**: Pace, pitch, expressivity, noise, crossfade
- 🖥️ **Gradio UI + FastAPI + CLI**

## Quick Start (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/beginner4a3/pod2/blob/main/colab_notebook.ipynb)

## Local Setup (Windows)

```powershell
cd c:\pod2\unified-podcast-generator

# 1. Set HuggingFace token (one-time for model download)
$env:HF_TOKEN = "your_hf_token_here"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the TTS model
huggingface-cli download ai4bharat/indic-parler-tts

# 4. Run the UI
python src/main.py --ui
```

## Usage

### Gradio UI (Recommended)

```bash
python src/main.py --ui
```

Features:
- Upload PDF/DOCX/TXT documents
- Generate scripts with LLM
- Edit scripts before generation
- Advanced audio settings
- Optional voice cloning per speaker

### FastAPI

```bash
uvicorn src.api:app --reload
```

Endpoints:
- `POST /upload` - Upload document
- `POST /generate-script` - Generate script
- `POST /generate-podcast` - Generate podcast
- `GET /speakers/{language}` - List speakers

### CLI

```bash
python src/main.py --text "नमस्ते" --speaker Rohit --output audio.wav
python src/main.py --list-languages
python src/main.py --list-speakers hindi
```

## Optional Features

### Voice Cloning (XTTS v2)

```bash
pip install TTS>=0.22.0
```

Upload a ~6 second reference audio per speaker for voice cloning.

### LLM Script Generation

```bash
pip install llama-cpp-python
```

Download a GGUF model (e.g., Mistral-7B) and provide the path in the UI.

## Model Info

| Property | Value |
|----------|-------|
| TTS Model | [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) |
| Languages | 21 |
| Voices | 69 |
| License | Apache 2.0 |

## Project Structure

```
src/
├── api.py              # FastAPI endpoints
├── main.py             # CLI entry point
├── tts/
│   ├── indic_parler.py # Main TTS engine
│   └── xtts_cloner.py  # Voice cloning (optional)
├── llm/
│   └── llama_local.py  # Self-hosted LLM
├── script/
│   └── document_parser.py # PDF/DOCX/TXT parsing
├── audio/
│   └── mixer.py        # Audio mixing
└── ui/
    └── gradio_app.py   # Gradio interface
```

## Credits

- **Indic-ParlerTTS**: AI4Bharat (IIT Madras) + HuggingFace
- **XTTS v2**: Coqui (for voice cloning)
- **Audio Processing**: Ported from kokoro-podcast-generator
