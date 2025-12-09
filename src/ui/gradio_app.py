"""
Unified Podcast Generator - Gradio UI

A beautiful, production-ready interface for generating podcasts
with Indic-ParlerTTS (21 Indian languages, 69 voices, 12 emotions).

Model: https://huggingface.co/ai4bharat/indic-parler-tts
"""

import gradio as gr
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tts.indic_parler import IndicParlerTTS, SPEAKERS, EMOTIONS

# Initialize TTS (lazy loading - model should be pre-installed)
tts_engine: Optional[IndicParlerTTS] = None


def get_tts() -> IndicParlerTTS:
    """Get or create TTS engine instance. Model must be pre-installed."""
    global tts_engine
    if tts_engine is None:
        tts_engine = IndicParlerTTS()
    return tts_engine


def get_speakers_for_language(language: str) -> list:
    """Get speaker names for a language."""
    speakers = SPEAKERS.get(language.lower(), [])
    return [s.name for s in speakers]


def update_speakers(language: str) -> Tuple[gr.Dropdown, gr.Dropdown]:
    """Update speaker dropdowns when language changes."""
    speakers = get_speakers_for_language(language)
    recommended = [s.name for s in SPEAKERS.get(language.lower(), []) if s.recommended]
    
    default1 = recommended[0] if len(recommended) > 0 else (speakers[0] if speakers else "Rohit")
    default2 = recommended[1] if len(recommended) > 1 else (speakers[1] if len(speakers) > 1 else "Divya")
    
    return (
        gr.Dropdown(choices=speakers, value=default1),
        gr.Dropdown(choices=speakers, value=default2)
    )


def generate_single_tts(
    text: str,
    speaker: str,
    emotion: str,
    progress=gr.Progress()
) -> Tuple[Tuple[int, np.ndarray], str]:
    """Generate TTS for a single text input."""
    if not text.strip():
        return None, "⚠️ Please enter some text."
    
    try:
        progress(0.2, desc="Loading model...")
        tts = get_tts()
        
        progress(0.5, desc="Generating speech...")
        audio = tts.generate(text, speaker=speaker, emotion=emotion)
        
        progress(1.0, desc="Done!")
        return (tts.sample_rate, audio), f"✅ Generated with {speaker}'s voice ({emotion})"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def generate_podcast(
    script: str,
    language: str,
    speaker1: str,
    speaker2: str,
    emotion1: str,
    emotion2: str,
    add_noise: bool,
    progress=gr.Progress()
) -> Tuple[Tuple[int, np.ndarray], str]:
    """Generate a full podcast from script."""
    if not script.strip():
        return None, "⚠️ Please enter a script."
    
    try:
        from src.audio.mixer import PodcastMixer, AudioClip
        
        progress(0.1, desc="Parsing script...")
        
        # Parse simple script format (Speaker1: text\nSpeaker2: text)
        lines = script.strip().split('\n')
        turns = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if ':' in line:
                speaker_part, text = line.split(':', 1)
                speaker_part = speaker_part.strip().lower()
                text = text.strip()
                
                if 'speaker1' in speaker_part or speaker1.lower() in speaker_part:
                    turns.append((speaker1, text, emotion1))
                elif 'speaker2' in speaker_part or speaker2.lower() in speaker_part:
                    turns.append((speaker2, text, emotion2))
                else:
                    # Alternate speakers
                    speaker = speaker1 if len(turns) % 2 == 0 else speaker2
                    emotion = emotion1 if len(turns) % 2 == 0 else emotion2
                    turns.append((speaker, text, emotion))
            else:
                # No speaker prefix, alternate
                speaker = speaker1 if len(turns) % 2 == 0 else speaker2
                emotion = emotion1 if len(turns) % 2 == 0 else emotion2
                turns.append((speaker, line, emotion))
        
        if not turns:
            return None, "⚠️ Could not parse script. Use format: 'Speaker1: text'"
        
        progress(0.2, desc="Loading TTS model...")
        tts = get_tts()
        
        # Generate audio for each turn
        audio_clips = []
        for i, (speaker, text, emotion) in enumerate(turns):
            progress(0.2 + (0.7 * i / len(turns)), desc=f"Generating turn {i+1}/{len(turns)}...")
            audio = tts.generate(text, speaker=speaker, emotion=emotion)
            audio_clips.append(AudioClip(audio=audio, sample_rate=tts.sample_rate, speaker=speaker, text=text))
        
        progress(0.9, desc="Mixing audio...")
        
        # Mix turns
        mixer = PodcastMixer(sample_rate=tts.sample_rate)
        final_audio = mixer.mix_turns(audio_clips, add_noise=add_noise)
        
        progress(1.0, desc="Done!")
        
        duration = len(final_audio) / tts.sample_rate
        return (tts.sample_rate, final_audio), f"✅ Generated {len(turns)} turns ({duration:.1f}s)"
        
    except Exception as e:
        import traceback
        return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"


# Build Gradio Interface
def create_interface():
    """Create the Gradio interface."""
    
    # Get initial values
    languages = list(SPEAKERS.keys())
    default_lang = "hindi"
    default_speakers = get_speakers_for_language(default_lang)
    
    with gr.Blocks(
        title="🎙️ Unified Podcast Generator",
        theme=gr.themes.Soft(),
        css="""
            .main-title { text-align: center; margin-bottom: 1em; }
            .info-box { padding: 1em; background: #f0f7ff; border-radius: 8px; margin: 1em 0; }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🎙️ Unified Podcast Generator
            
            Generate podcasts in **21 Indian languages** using **Indic-ParlerTTS** with **69 voices** and **12 emotions**.
            
            **Supported Languages**: Hindi, Telugu, Tamil, Malayalam, Kannada, Bengali, Marathi, Gujarati, Assamese, Odia, Punjabi, Nepali, and more!
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Tabs():
            # Tab 1: Single TTS
            with gr.TabItem("🎤 Single TTS"):
                gr.Markdown("### Generate speech from text")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        single_text = gr.Textbox(
                            label="Text to speak",
                            placeholder="नमस्ते, आज हम AI के बारे में बात करेंगे",
                            lines=3
                        )
                        
                        with gr.Row():
                            single_language = gr.Dropdown(
                                choices=languages,
                                value=default_lang,
                                label="Language"
                            )
                            single_speaker = gr.Dropdown(
                                choices=default_speakers,
                                value="Rohit",
                                label="Speaker"
                            )
                            single_emotion = gr.Dropdown(
                                choices=EMOTIONS,
                                value="neutral",
                                label="Emotion"
                            )
                        
                        single_btn = gr.Button("🎙️ Generate", variant="primary")
                        
                    with gr.Column(scale=1):
                        single_audio = gr.Audio(label="Generated Audio", type="numpy")
                        single_status = gr.Textbox(label="Status", interactive=False)
                
                # Update speakers when language changes
                single_language.change(
                    fn=lambda lang: gr.Dropdown(choices=get_speakers_for_language(lang)),
                    inputs=[single_language],
                    outputs=[single_speaker]
                )
                
                single_btn.click(
                    fn=generate_single_tts,
                    inputs=[single_text, single_speaker, single_emotion],
                    outputs=[single_audio, single_status]
                )
            
            # Tab 2: Podcast Generator
            with gr.TabItem("🎬 Podcast"):
                gr.Markdown("### Generate a podcast from script")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        podcast_script = gr.Textbox(
                            label="Podcast Script",
                            placeholder="""Speaker1: नमस्ते दोस्तों! आज हम AI के बारे में बात करेंगे।
Speaker2: हाँ, यह बहुत interesting topic है।
Speaker1: AI का इस्तेमाल आजकल हर जगह हो रहा है।
Speaker2: Medical field में इसका बहुत फायदा हो रहा है।""",
                            lines=10
                        )
                        
                        podcast_language = gr.Dropdown(
                            choices=languages,
                            value=default_lang,
                            label="Language"
                        )
                        
                        with gr.Row():
                            podcast_speaker1 = gr.Dropdown(
                                choices=default_speakers,
                                value="Rohit",
                                label="Speaker 1"
                            )
                            podcast_emotion1 = gr.Dropdown(
                                choices=EMOTIONS,
                                value="conversation",
                                label="Emotion 1"
                            )
                        
                        with gr.Row():
                            podcast_speaker2 = gr.Dropdown(
                                choices=default_speakers,
                                value="Divya",
                                label="Speaker 2"
                            )
                            podcast_emotion2 = gr.Dropdown(
                                choices=EMOTIONS,
                                value="conversation",
                                label="Emotion 2"
                            )
                        
                        podcast_noise = gr.Checkbox(
                            label="Add background noise (for realism)",
                            value=True
                        )
                        
                        podcast_btn = gr.Button("🎬 Generate Podcast", variant="primary")
                        
                    with gr.Column(scale=1):
                        podcast_audio = gr.Audio(label="Generated Podcast", type="numpy")
                        podcast_status = gr.Textbox(label="Status", interactive=False)
                
                # Update speakers when language changes
                podcast_language.change(
                    fn=update_speakers,
                    inputs=[podcast_language],
                    outputs=[podcast_speaker1, podcast_speaker2]
                )
                
                podcast_btn.click(
                    fn=generate_podcast,
                    inputs=[
                        podcast_script, podcast_language,
                        podcast_speaker1, podcast_speaker2,
                        podcast_emotion1, podcast_emotion2,
                        podcast_noise
                    ],
                    outputs=[podcast_audio, podcast_status]
                )
            
            # Tab 3: Info
            with gr.TabItem("ℹ️ Info"):
                gr.Markdown(
                    """
                    ## About
                    
                    This project merges three podcast generators:
                    - **kokoro-podcast-generator** (audio mixing)
                    - **Podcastfy.ai_demo** (UI patterns)
                    - **Voice-Clone-Podcast** (multi-language prompts)
                    
                    ## TTS Engine
                    
                    **Indic-ParlerTTS** by AI4Bharat (IIT Madras) + HuggingFace
                    - Model: [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)
                    - License: Apache 2.0
                    
                    ## Languages (21)
                    
                    Hindi, Telugu, Tamil, Malayalam, Kannada, Bengali, Marathi, Gujarati, 
                    Assamese, Odia, Punjabi, Nepali, Sanskrit, Bodo, Dogri, Konkani, 
                    Maithili, Manipuri, Santali, Sindhi, Urdu, English
                    
                    ## Emotions (12)
                    
                    Neutral, Happy, Sad, Anger, Fear, Surprise, Disgust, 
                    Command, Narration, Conversation, News, Proper Noun
                    """
                )
        
        gr.Markdown(
            """
            ---
            Made with ❤️ using Indic-ParlerTTS | [HuggingFace](https://huggingface.co/ai4bharat/indic-parler-tts)
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.queue().launch(share=False)
