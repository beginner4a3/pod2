# Unified Podcast Generator

AI Podcast Generator with **Indic-ParlerTTS** (21 languages, 69 voices) + **LLM Script Generation** + **Voice Cloning**

## Quick Start

### Local PC (Windows/Linux)

```bash
cd c:\pod2\unified-podcast-generator

# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download models (TTS + LLM = ~7GB)
python setup_models.py

# Step 3: Launch UI
python src/main.py --ui
```

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/beginner4a3/pod2/blob/main/colab_notebook.ipynb)

## Features

- 🎙️ **21 Indian Languages**: Hindi, Telugu, Tamil, Malayalam, Kannada, Bengali, etc.
- 🗣️ **69 Voices**: Male/Female speakers with 12 emotions
- 🤖 **LLM Script Generation**: BharatGPT-3B-Indic (optimized for Indian languages)
- 📄 **Document Upload**: PDF, DOCX, TXT
- 🎤 **Voice Cloning**: XTTS v2 (upload ~6 sec reference audio)
- 🎛️ **Advanced Settings**: Pace, pitch, expressivity, noise, crossfade

## Usage

### Gradio UI

```bash
python src/main.py --ui
```

### FastAPI

```bash
uvicorn src.api:app --reload
```

### CLI

```bash
python src/main.py --text "नमस्ते" --speaker Rohit --output audio.wav
```

## Project Structure

```
src/
├── api.py              # FastAPI endpoints
├── main.py             # CLI entry point
├── tts/
│   ├── indic_parler.py # Main TTS (Indic-ParlerTTS)
│   └── xtts_cloner.py  # Voice cloning (XTTS v2)
├── llm/
│   └── llama_local.py  # LLM script generation
├── script/
│   └── document_parser.py
├── audio/
│   └── mixer.py
└── ui/
    └── gradio_app.py

setup_models.py         # Downloads all models
requirements.txt        # All dependencies
```

## Credits

- **TTS**: [AI4Bharat Indic-ParlerTTS](https://huggingface.co/ai4bharat/indic-parler-tts)
- **Voice Cloning**: [Coqui XTTS v2](https://huggingface.co/coqui/XTTS-v2)
- **LLM**: [BharatGPT-3B-Indic](https://huggingface.co/QuantFactory/BharatGPT-3B-Indic-GGUF)
