"""
Quick Test Script for Indic-ParlerTTS

Tests if the pre-downloaded TTS model works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 60)
    print("🎙️ Indic-ParlerTTS Quick Test")
    print("=" * 60)
    
    # Import TTS
    print("\n📥 Importing IndicParlerTTS...")
    from src.tts.indic_parler import IndicParlerTTS
    
    # Check available languages and speakers
    print(f"\n📋 Supported languages: {', '.join(IndicParlerTTS.get_languages())}")
    print(f"📋 Hindi speakers: {[s.name for s in IndicParlerTTS.get_speakers('hindi')]}")
    print(f"📋 Emotions: {', '.join(IndicParlerTTS.get_emotions())}")
    
    # Create TTS instance
    print("\n🔧 Creating TTS instance...")
    tts = IndicParlerTTS()
    
    # Load model
    print("\n📥 Loading model (first run may take a few minutes)...")
    print(f"   Model: {tts.MODEL_URL}")
    tts.load()
    
    # Generate test audio
    print("\n🎤 Generating test audio...")
    test_text = "नमस्ते, यह एक परीक्षण है।"
    print(f"   Text: {test_text}")
    print(f"   Speaker: Rohit")
    print(f"   Emotion: happy")
    
    audio = tts.generate(test_text, speaker="Rohit", emotion="happy")
    
    # Save
    output_path = Path(__file__).parent / "test_output.wav"
    tts.save(audio, str(output_path))
    
    duration = len(audio) / tts.sample_rate
    print(f"\n✅ Success! Audio saved to: {output_path}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Sample rate: {tts.sample_rate} Hz")
    
    print("\n" + "=" * 60)
    print("🎉 Test complete! Everything is working.")
    print("=" * 60)


if __name__ == "__main__":
    main()
