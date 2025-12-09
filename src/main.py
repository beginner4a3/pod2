"""
Unified Podcast Generator - Main Entry Point

Usage:
    # Run Gradio UI
    python src/main.py --ui
    
    # Generate from CLI
    python src/main.py --text "नमस्ते" --speaker Rohit --output output.wav
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Unified Podcast Generator with Indic-ParlerTTS"
    )
    
    # Mode selection
    parser.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    
    # TTS options
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--speaker", type=str, default="Rohit", help="Speaker name")
    parser.add_argument("--emotion", type=str, default="neutral", help="Emotion")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file path")
    
    # List options
    parser.add_argument("--list-languages", action="store_true", help="List supported languages")
    parser.add_argument("--list-speakers", type=str, help="List speakers for a language")
    parser.add_argument("--list-emotions", action="store_true", help="List supported emotions")
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_languages:
        from src.tts.indic_parler import IndicParlerTTS
        print("Supported Languages:")
        for lang in IndicParlerTTS.get_languages():
            print(f"  - {lang}")
        return
    
    if args.list_speakers:
        from src.tts.indic_parler import IndicParlerTTS
        speakers = IndicParlerTTS.get_speakers(args.list_speakers)
        if speakers:
            print(f"Speakers for {args.list_speakers}:")
            for s in speakers:
                rec = " (recommended)" if s.recommended else ""
                print(f"  - {s.name} ({s.gender}){rec}")
        else:
            print(f"No speakers found for language: {args.list_speakers}")
        return
    
    if args.list_emotions:
        from src.tts.indic_parler import IndicParlerTTS
        print("Supported Emotions:")
        for emotion in IndicParlerTTS.get_emotions():
            print(f"  - {emotion}")
        return
    
    # Launch UI
    if args.ui:
        from src.ui.gradio_app import create_interface
        demo = create_interface()
        demo.queue().launch(share=False)
        return
    
    # Generate TTS
    if args.text:
        from src.tts.indic_parler import IndicParlerTTS
        
        print(f"Generating speech for: {args.text[:50]}...")
        print(f"Speaker: {args.speaker}, Emotion: {args.emotion}")
        
        tts = IndicParlerTTS()
        audio = tts.generate(args.text, speaker=args.speaker, emotion=args.emotion)
        tts.save(audio, args.output)
        
        print(f"✅ Saved to: {args.output}")
        return
    
    # No command - show help
    parser.print_help()


if __name__ == "__main__":
    main()
