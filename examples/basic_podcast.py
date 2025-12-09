"""
Basic Podcast Generation Example

This example shows how to:
1. Generate TTS for multiple turns
2. Mix them into a podcast
3. Save as WAV or MP3
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tts.indic_parler import IndicParlerTTS
from src.audio.mixer import PodcastMixer, AudioClip
from src.audio.formats import save_wav, save_mp3


def main():
    # Script for the podcast (Hindi)
    script = [
        ("Rohit", "नमस्ते दोस्तों! आज के podcast में हम Artificial Intelligence के बारे में बात करेंगे।", "happy"),
        ("Divya", "हाँ Rohit, यह बहुत interesting topic है। AI आजकल हर जगह इस्तेमाल हो रहा है।", "conversation"),
        ("Rohit", "बिल्कुल सही कहा। Healthcare में AI का बहुत बड़ा impact है।", "neutral"),
        ("Divya", "Medical imaging में AI doctors को diagnosis में मदद कर रहा है।", "conversation"),
        ("Rohit", "और education में भी AI personalized learning को possible बना रहा है।", "happy"),
        ("Divya", "Thanks for listening everyone! अगले episode में फिर मिलेंगे।", "happy"),
    ]
    
    print("🎙️ Unified Podcast Generator - Basic Example")
    print("=" * 50)
    
    # Initialize TTS
    print("\n📥 Loading Indic-ParlerTTS model...")
    tts = IndicParlerTTS()
    
    # Generate audio for each turn
    print("\n🎤 Generating speech for each turn...")
    audio_clips = []
    
    for i, (speaker, text, emotion) in enumerate(script):
        print(f"  [{i+1}/{len(script)}] {speaker}: {text[:30]}...")
        audio = tts.generate(text, speaker=speaker, emotion=emotion)
        audio_clips.append(AudioClip(
            audio=audio,
            sample_rate=tts.sample_rate,
            speaker=speaker,
            text=text
        ))
    
    # Mix into podcast
    print("\n🎛️ Mixing podcast...")
    mixer = PodcastMixer(sample_rate=tts.sample_rate)
    final_audio = mixer.mix_turns(
        audio_clips,
        gap_ms=200,  # 200ms gap between turns
        add_noise=True,
        noise_level=0.002
    )
    
    # Save
    output_path = Path(__file__).parent / "output_podcast.wav"
    save_wav(final_audio, tts.sample_rate, str(output_path))
    
    duration = len(final_audio) / tts.sample_rate
    print(f"\n✅ Podcast saved to: {output_path}")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Turns: {len(script)}")


if __name__ == "__main__":
    main()
