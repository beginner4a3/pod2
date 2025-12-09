"""
Unified Podcast Generator - Redesigned Gradio UI

Features:
- File upload (PDF/DOCX/TXT)
- LLM-powered script generation
- Editable script textarea
- Advanced audio settings
- Optional voice cloning per speaker

Model: https://huggingface.co/ai4bharat/indic-parler-tts
"""

import gradio as gr
import numpy as np
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tts.indic_parler import IndicParlerTTS, SPEAKERS, EMOTIONS, TTSConfig
from src.script.document_parser import parse_document
from src.tts.xtts_cloner import is_voice_cloning_available

# Initialize TTS (lazy loading)
tts_engine: Optional[IndicParlerTTS] = None
xtts_engine = None


def get_tts() -> IndicParlerTTS:
    """Get or create TTS engine instance."""
    global tts_engine
    if tts_engine is None:
        tts_engine = IndicParlerTTS()
    return tts_engine


def get_xtts():
    """Get or create XTTS engine for voice cloning."""
    global xtts_engine
    if xtts_engine is None and is_voice_cloning_available():
        from src.tts.xtts_cloner import XTTSCloner
        xtts_engine = XTTSCloner()
    return xtts_engine


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


def extract_document_text(file) -> Tuple[str, str]:
    """Extract text from uploaded document."""
    if file is None:
        return "", "⚠️ No file uploaded"
    
    try:
        result = parse_document(file.name)
        preview = result["text"][:2000] + "..." if len(result["text"]) > 2000 else result["text"]
        status = f"✅ Extracted {result['word_count']} words from {result['file_name']}"
        return preview, status
    except Exception as e:
        return "", f"❌ Error: {str(e)}"


def generate_script_from_content(
    content: str,
    language: str,
    style: str,
    num_turns: int,
    model_path: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate podcast script from content using LLM."""
    if not content.strip():
        return "", "⚠️ No content provided"
    
    if not model_path or not Path(model_path).exists():
        return "", "⚠️ LLM model not found. Please provide a valid GGUF model path."
    
    try:
        progress(0.2, desc="Loading LLM...")
        from src.llm.llama_local import LlamaLocalLLM
        
        llm = LlamaLocalLLM()
        llm.load(model_path)
        
        progress(0.5, desc="Generating script...")
        result = llm.generate_podcast_script(
            topic=content[:4000],  # Limit content size
            language=language,
            num_turns=num_turns,
            style=style
        )
        
        progress(1.0, desc="Done!")
        
        if "raw_response" in result:
            return result["raw_response"], "✅ Script generated (raw format)"
        
        # Format structured response
        turns = result.get("turns", [])
        script_lines = []
        for turn in turns:
            speaker = turn.get("speaker", "Speaker1")
            text = turn.get("text", "")
            script_lines.append(f"{speaker}: {text}")
        
        script = "\n".join(script_lines)
        title = result.get("title", "Untitled")
        
        return script, f"✅ Generated: {title} ({len(turns)} turns)"
        
    except Exception as e:
        return "", f"❌ Error: {str(e)}"


def generate_podcast(
    script: str,
    language: str,
    speaker1_name: str,
    speaker1_emotion: str,
    speaker1_ref: Optional[str],
    speaker2_name: str,
    speaker2_emotion: str,
    speaker2_ref: Optional[str],
    pace: str,
    pitch: str,
    expressivity: str,
    gap_ms: int,
    crossfade: bool,
    intro_silence_ms: int,
    add_noise: bool,
    noise_level: float,
    normalize: bool,
    progress=gr.Progress()
) -> Tuple[Tuple[int, np.ndarray], str]:
    """Generate podcast from script with all settings."""
    if not script.strip():
        return None, "⚠️ Please enter a script"
    
    try:
        from src.audio.mixer import PodcastMixer, AudioClip, add_silence, normalize_volume
        
        progress(0.1, desc="Parsing script...")
        
        # Parse script
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
                
                if 'speaker1' in speaker_part or speaker1_name.lower() in speaker_part:
                    turns.append((1, text))
                elif 'speaker2' in speaker_part or speaker2_name.lower() in speaker_part:
                    turns.append((2, text))
                else:
                    speaker_num = 1 if len(turns) % 2 == 0 else 2
                    turns.append((speaker_num, text))
            else:
                speaker_num = 1 if len(turns) % 2 == 0 else 2
                turns.append((speaker_num, line))
        
        if not turns:
            return None, "⚠️ Could not parse script. Use format: 'Speaker1: text'"
        
        progress(0.2, desc="Initializing TTS...")
        tts = get_tts()
        xtts = get_xtts()
        
        # Check voice cloning availability
        use_clone_s1 = speaker1_ref is not None and xtts is not None
        use_clone_s2 = speaker2_ref is not None and xtts is not None
        
        # Generate audio for each turn
        audio_clips = []
        config = TTSConfig(
            pace=pace,
            pitch=pitch,
            expressivity=expressivity
        )
        
        for i, (speaker_num, text) in enumerate(turns):
            progress(0.2 + (0.6 * i / len(turns)), desc=f"Generating turn {i+1}/{len(turns)}...")
            
            if speaker_num == 1:
                speaker_name = speaker1_name
                emotion = speaker1_emotion
                use_clone = use_clone_s1
                ref_audio = speaker1_ref
            else:
                speaker_name = speaker2_name
                emotion = speaker2_emotion
                use_clone = use_clone_s2
                ref_audio = speaker2_ref
            
            if use_clone and ref_audio:
                # Use XTTS for cloned voice
                audio = xtts.generate(text, ref_audio, language)
            else:
                # Use Indic-ParlerTTS
                config.emotion = emotion
                audio = tts.generate(text, speaker=speaker_name, config=config)
            
            audio_clips.append(AudioClip(
                audio=audio, 
                sample_rate=tts.sample_rate, 
                speaker=speaker_name, 
                text=text
            ))
        
        progress(0.85, desc="Mixing audio...")
        
        # Mix turns
        mixer = PodcastMixer(sample_rate=tts.sample_rate)
        final_audio = mixer.mix_turns(
            audio_clips, 
            gap_ms=gap_ms, 
            add_noise=add_noise, 
            noise_level=noise_level
        )
        
        # Add intro silence
        if intro_silence_ms > 0:
            final_audio = add_silence(final_audio, intro_silence_ms, tts.sample_rate, at_beginning=True)
        
        # Normalize
        if normalize:
            final_audio = normalize_volume(final_audio)
        
        progress(1.0, desc="Done!")
        
        duration = len(final_audio) / tts.sample_rate
        clone_status = ""
        if use_clone_s1:
            clone_status += f" | {speaker1_name}: 🎤 cloned"
        if use_clone_s2:
            clone_status += f" | {speaker2_name}: 🎤 cloned"
        
        status = f"✅ Generated {len(turns)} turns ({duration:.1f}s){clone_status}"
        
        return (tts.sample_rate, final_audio), status
        
    except Exception as e:
        import traceback
        return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# Build Gradio Interface
# ============================================================================

def create_interface():
    """Create the Gradio interface."""
    
    languages = list(SPEAKERS.keys())
    default_lang = "hindi"
    default_speakers = get_speakers_for_language(default_lang)
    voice_cloning_available = is_voice_cloning_available()
    
    with gr.Blocks(
        title="🎙️ Unified Podcast Generator",
        theme=gr.themes.Soft(),
        css="""
            .main-title { text-align: center; margin-bottom: 0.5em; }
            .section-title { margin-top: 1em; font-weight: bold; }
            .compact-row { gap: 0.5em; }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🎙️ Unified Podcast Generator
            
            Generate podcasts in **21 Indian languages** using **Indic-ParlerTTS** (69 voices, 12 emotions).
            Upload a document, generate a script with AI, and create your podcast!
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Tabs():
            # ================================================================
            # Tab 1: Podcast Generator (Main)
            # ================================================================
            with gr.TabItem("🎬 Generate Podcast"):
                
                with gr.Row():
                    # Left Column - Input
                    with gr.Column(scale=2):
                        
                        # --- Content Source ---
                        gr.Markdown("### 📄 Content Source")
                        
                        with gr.Tabs():
                            with gr.TabItem("📁 Upload File"):
                                file_upload = gr.File(
                                    label="Upload Document (PDF, DOCX, TXT)",
                                    file_types=[".pdf", ".docx", ".txt"]
                                )
                                extract_btn = gr.Button("📤 Extract Text", variant="secondary")
                            
                            with gr.TabItem("✍️ Type/Paste"):
                                pass  # Content goes to main textarea
                        
                        content_text = gr.Textbox(
                            label="Content / Topic",
                            placeholder="Paste your content here, or upload a file above...\n\nExample: Write about the future of AI in India",
                            lines=6
                        )
                        extract_status = gr.Textbox(label="Status", interactive=False, visible=True)
                        
                        # --- Script Generation ---
                        gr.Markdown("### 🤖 Script Generation")
                        
                        with gr.Row():
                            script_language = gr.Dropdown(
                                choices=languages,
                                value=default_lang,
                                label="Language"
                            )
                            script_style = gr.Dropdown(
                                choices=["conversational", "news", "storytelling", "interview"],
                                value="conversational",
                                label="Style"
                            )
                            num_turns = gr.Slider(
                                minimum=5, maximum=30, value=15, step=1,
                                label="Turns"
                            )
                        
                        with gr.Row():
                            llm_model_path = gr.Textbox(
                                label="LLM Model Path (GGUF)",
                                placeholder="path/to/mistral-7b.gguf",
                                scale=3
                            )
                            generate_script_btn = gr.Button("🤖 Generate Script", variant="secondary")
                        
                        # --- Editable Script ---
                        podcast_script = gr.Textbox(
                            label="📝 Podcast Script (Editable)",
                            placeholder="""Speaker1: नमस्ते दोस्तों! आज हम AI के बारे में बात करेंगे।
Speaker2: हाँ, यह बहुत interesting topic है।
Speaker1: AI का इस्तेमाल आजकल हर जगह हो रहा है।
Speaker2: Medical field में इसका बहुत फायदा हो रहा है।""",
                            lines=10
                        )
                        script_status = gr.Textbox(label="Script Status", interactive=False)
                        
                        # --- Speakers ---
                        gr.Markdown("### 👥 Speakers")
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Speaker 1**")
                                speaker1_name = gr.Dropdown(
                                    choices=default_speakers,
                                    value="Rohit",
                                    label="Name"
                                )
                                speaker1_emotion = gr.Dropdown(
                                    choices=EMOTIONS,
                                    value="conversation",
                                    label="Emotion"
                                )
                                if voice_cloning_available:
                                    speaker1_ref = gr.Audio(
                                        label="🎤 Reference Voice (optional)",
                                        type="filepath"
                                    )
                                else:
                                    speaker1_ref = gr.State(None)
                            
                            with gr.Column():
                                gr.Markdown("**Speaker 2**")
                                speaker2_name = gr.Dropdown(
                                    choices=default_speakers,
                                    value="Divya",
                                    label="Name"
                                )
                                speaker2_emotion = gr.Dropdown(
                                    choices=EMOTIONS,
                                    value="conversation",
                                    label="Emotion"
                                )
                                if voice_cloning_available:
                                    speaker2_ref = gr.Audio(
                                        label="🎤 Reference Voice (optional)",
                                        type="filepath"
                                    )
                                else:
                                    speaker2_ref = gr.State(None)
                        
                        # --- Advanced Settings ---
                        with gr.Accordion("🎛️ Advanced Audio Settings", open=False):
                            
                            gr.Markdown("**TTS Settings**")
                            with gr.Row():
                                tts_pace = gr.Dropdown(
                                    choices=["slow", "moderate", "fast"],
                                    value="moderate",
                                    label="Pace"
                                )
                                tts_pitch = gr.Dropdown(
                                    choices=["low", "moderate", "high"],
                                    value="moderate",
                                    label="Pitch"
                                )
                                tts_expressivity = gr.Dropdown(
                                    choices=["monotone", "slightly expressive", "very expressive"],
                                    value="slightly expressive",
                                    label="Expressivity"
                                )
                            
                            gr.Markdown("**Mixing Settings**")
                            with gr.Row():
                                gap_ms = gr.Slider(
                                    minimum=0, maximum=500, value=100, step=10,
                                    label="Gap between turns (ms)"
                                )
                                crossfade = gr.Checkbox(label="Crossfade", value=False)
                                intro_silence = gr.Slider(
                                    minimum=0, maximum=2000, value=500, step=100,
                                    label="Intro silence (ms)"
                                )
                            
                            gr.Markdown("**Post-Processing**")
                            with gr.Row():
                                add_noise = gr.Checkbox(label="Add background noise", value=True)
                                noise_level = gr.Slider(
                                    minimum=0.001, maximum=0.01, value=0.002, step=0.001,
                                    label="Noise level"
                                )
                                normalize = gr.Checkbox(label="Normalize volume", value=True)
                        
                        # --- Generate Button ---
                        generate_podcast_btn = gr.Button(
                            "🎬 Generate Podcast",
                            variant="primary",
                            size="lg"
                        )
                    
                    # Right Column - Output
                    with gr.Column(scale=1):
                        podcast_audio = gr.Audio(
                            label="🎧 Generated Podcast",
                            type="numpy"
                        )
                        podcast_status = gr.Textbox(label="Status", interactive=False)
                        
                        if not voice_cloning_available:
                            gr.Markdown(
                                """
                                > 💡 **Voice Cloning**: Install `TTS` library for optional voice cloning:
                                > `pip install TTS>=0.22.0`
                                """
                            )
                
                # --- Event Handlers ---
                
                # Update speakers when language changes
                script_language.change(
                    fn=update_speakers,
                    inputs=[script_language],
                    outputs=[speaker1_name, speaker2_name]
                )
                
                # Extract text from file
                extract_btn.click(
                    fn=extract_document_text,
                    inputs=[file_upload],
                    outputs=[content_text, extract_status]
                )
                
                # Generate script
                generate_script_btn.click(
                    fn=generate_script_from_content,
                    inputs=[
                        content_text, script_language, script_style, 
                        num_turns, llm_model_path
                    ],
                    outputs=[podcast_script, script_status]
                )
                
                # Generate podcast
                generate_podcast_btn.click(
                    fn=generate_podcast,
                    inputs=[
                        podcast_script, script_language,
                        speaker1_name, speaker1_emotion, speaker1_ref,
                        speaker2_name, speaker2_emotion, speaker2_ref,
                        tts_pace, tts_pitch, tts_expressivity,
                        gap_ms, crossfade, intro_silence,
                        add_noise, noise_level, normalize
                    ],
                    outputs=[podcast_audio, podcast_status]
                )
            
            # ================================================================
            # Tab 2: Info
            # ================================================================
            with gr.TabItem("ℹ️ Info"):
                gr.Markdown(
                    f"""
                    ## About
                    
                    This tool generates AI podcasts using Indic-ParlerTTS.
                    
                    ## TTS Engine
                    
                    **Indic-ParlerTTS** by AI4Bharat (IIT Madras)
                    - Model: [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)
                    - Languages: 21 | Voices: 69 | Emotions: 12
                    
                    ## Voice Cloning
                    
                    **Status**: {"✅ Available (XTTS v2)" if voice_cloning_available else "❌ Not installed"}
                    
                    Voice cloning uses Coqui XTTS v2 and requires:
                    - `pip install TTS>=0.22.0`
                    - ~6 second reference audio per speaker
                    
                    ## Languages
                    
                    Hindi, Telugu, Tamil, Malayalam, Kannada, Bengali, Marathi, Gujarati, 
                    Assamese, Odia, Punjabi, Nepali, Sanskrit, Bodo, Dogri, Konkani, 
                    Maithili, Manipuri, Santali, Sindhi, Urdu, English
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
