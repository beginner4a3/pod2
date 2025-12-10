"""
Setup Script - Downloads all required models

Run this after pip install -r requirements.txt:
    python setup_models.py

Downloads:
1. Indic-ParlerTTS model (~3GB)
2. BharatGPT-3B-Indic LLM (~2GB) - Optimized for Indian languages!
"""

import os
import sys
from pathlib import Path


# LLM Options (choose one)
LLM_OPTIONS = {
    "bharatgpt": {
        "repo": "QuantFactory/BharatGPT-3B-Indic-GGUF",
        "file": "BharatGPT-3B-Indic.Q4_K_M.gguf",
        "size": "~2GB",
        "desc": "BharatGPT-3B (BEST for Hindi/Telugu/Indian languages)"
    },
    "mistral": {
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size": "~4GB",
        "desc": "Mistral-7B (Better for English)"
    }
}

# Default: BharatGPT for Indian languages
DEFAULT_LLM = "bharatgpt"


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
    # 2. Download LLM Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("📥 [2/2] Downloading LLM Model...")
    print("=" * 60)
    
    # Choose LLM
    llm_choice = DEFAULT_LLM
    print("\nAvailable LLM options:")
    for key, opt in LLM_OPTIONS.items():
        marker = "→" if key == DEFAULT_LLM else " "
        print(f"  {marker} {key}: {opt['desc']} ({opt['size']})")
    
    user_choice = input(f"\nChoose LLM [{DEFAULT_LLM}]: ").strip().lower()
    if user_choice in LLM_OPTIONS:
        llm_choice = user_choice
    
    llm_config = LLM_OPTIONS[llm_choice]
    llm_filename = llm_config["file"]
    llm_path = models_dir / llm_filename
    
    print(f"\nDownloading: {llm_config['desc']}")
    
    if llm_path.exists():
        print(f"✅ LLM Model already exists: {llm_path}")
    else:
        try:
            downloaded_path = hf_hub_download(
                repo_id=llm_config["repo"],
                filename=llm_filename,
                local_dir=models_dir
            )
            print(f"✅ LLM Model: {downloaded_path}")
        except Exception as e:
            print(f"❌ LLM download failed: {e}")
            # Try alternate filename for BharatGPT
            if llm_choice == "bharatgpt":
                print("Trying alternate quantization...")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=llm_config["repo"],
                        filename="BharatGPT-3B-Indic.Q5_K_M.gguf",
                        local_dir=models_dir
                    )
                    print(f"✅ LLM Model: {downloaded_path}")
                except Exception as e2:
                    print(f"❌ Alternate also failed: {e2}")
    
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
