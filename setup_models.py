"""
Setup Script - Downloads all required models

Run this after pip install -r requirements.txt:
    python setup_models.py

Downloads:
1. Indic-ParlerTTS model (~3GB)
2. Mistral-7B LLM model (~4GB)
"""

import os
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("🎙️ Unified Podcast Generator - Model Setup")
    print("=" * 60)
    
    # Check HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n⚠️ HF_TOKEN not set!")
        print("Set it with: $env:HF_TOKEN = 'your_token_here' (PowerShell)")
        print("Or: set HF_TOKEN=your_token_here (CMD)")
        print("\nGet token from: https://huggingface.co/settings/tokens")
        
        # Ask for token
        hf_token = input("\nEnter your HuggingFace token (or press Enter to skip TTS model): ").strip()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
    
    # Import after setting token
    from huggingface_hub import snapshot_download, hf_hub_download
    
    models_dir = Path.cwd() / "models"
    models_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # 1. Download Indic-ParlerTTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("📥 [1/2] Downloading Indic-ParlerTTS (~3GB)...")
    print("=" * 60)
    
    if hf_token:
        try:
            tts_path = snapshot_download(
                repo_id="ai4bharat/indic-parler-tts",
                token=hf_token
            )
            print(f"✅ TTS Model: {tts_path}")
        except Exception as e:
            print(f"❌ TTS download failed: {e}")
    else:
        print("⏭️ Skipped (no token)")
    
    # =========================================================================
    # 2. Download LLM Model (Mistral-7B)
    # =========================================================================
    print("\n" + "=" * 60)
    print("📥 [2/2] Downloading Mistral-7B LLM (~4GB)...")
    print("=" * 60)
    
    llm_filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    llm_path = models_dir / llm_filename
    
    if llm_path.exists():
        print(f"✅ LLM Model already exists: {llm_path}")
    else:
        try:
            downloaded_path = hf_hub_download(
                repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                filename=llm_filename,
                local_dir=models_dir
            )
            print(f"✅ LLM Model: {downloaded_path}")
        except Exception as e:
            print(f"❌ LLM download failed: {e}")
    
    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("\nTo start the app:")
    print("  python src/main.py --ui")
    print("\nOr for Colab, use: colab_notebook.ipynb")


if __name__ == "__main__":
    main()
