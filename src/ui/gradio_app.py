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
    """Get speaker names for a language with style info."""
    speakers = SPEAKERS.get(language.lower(), [])
    # Format: "Name (Gender) - Style" or just "Name (Gender)"
    result = []
    for s in speakers:
        if s.style:
            result.append(f"{s.name} ({s.gender}) - {s.style}")
        else:
            result.append(f"{s.name} ({s.gender})")
    return result


def get_speaker_name_from_label(label: str) -> str:
    """Extract speaker name from dropdown label like 'Rohit (male) - Clear'."""
    if "(" in label:
        return label.split("(")[0].strip()
    return label


def update_speakers(language: str) -> Tuple[gr.Dropdown, gr.Dropdown]:
    """Update speaker dropdowns when language changes."""
    speaker_labels = get_speakers_for_language(language)
    speaker_objs = SPEAKERS.get(language.lower(), [])
    
    # Find recommended speakers with full labels
    recommended_labels = []
    for s in speaker_objs:
        if s.recommended:
            if s.style:
                recommended_labels.append(f"{s.name} ({s.gender}) - {s.style}")
            else:
                recommended_labels.append(f"{s.name} ({s.gender})")
    
    default1 = recommended_labels[0] if len(recommended_labels) > 0 else (speaker_labels[0] if speaker_labels else "Rohit (male)")
    default2 = recommended_labels[1] if len(recommended_labels) > 1 else (speaker_labels[1] if len(speaker_labels) > 1 else "Divya (female)")
    
    return (
        gr.Dropdown(choices=speaker_labels, value=default1),
        gr.Dropdown(choices=speaker_labels, value=default2)
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
    gemini_api_key: str = "",
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Generate podcast script from content using Gemini API or local LLM."""
    if not content.strip():
        return "", "⚠️ No content provided"
    
    try:
        # Try Gemini API first (best quality)
        from src.llm.gemini_api import GeminiScriptGenerator
        import os
        
        # Use provided API key or environment variable
        api_key = gemini_api_key.strip() if gemini_api_key else os.environ.get("GEMINI_API_KEY", "")
        
        if api_key:
            progress(0.2, desc="Using Gemini API (best quality)...")
            generator = GeminiScriptGenerator(api_key=api_key)
            script = generator.generate_script(
                topic=content[:4000],
                language=language,
                num_turns=num_turns
            )
            progress(1.0, desc="Done!")
            return script, f"✅ Generated {len(script.splitlines())} turns using Gemini API"
        
        # Fallback to local LLM
        progress(0.1, desc="Loading local LLM...")
        from src.llm.llama_local import LlamaLocalLLM
        
        llm = LlamaLocalLLM()
        llm.load()  # Auto-detects pre-installed model
        
        progress(0.5, desc="Generating script (local LLM)...")
        result = llm.generate_podcast_script(
            topic=content[:4000],  # Limit content size
            language=language,
            num_turns=num_turns,
            style=style
        )
        
        progress(0.9, desc="Formatting script...")
        
        # Convert to proper Speaker1/Speaker2 format
        script_lines = []
        
        if "turns" in result:
            # JSON format with turns array
            for i, turn in enumerate(result.get("turns", [])):
                text = turn.get("text", "")
                # Normalize speaker name to Speaker1/Speaker2
                original_speaker = turn.get("speaker", "")
                if "1" in str(original_speaker) or i % 2 == 0:
                    speaker = "Speaker1"
                else:
                    speaker = "Speaker2"
                if text.strip():
                    script_lines.append(f"{speaker}: {text.strip()}")
            
            title = result.get("title", "Untitled")
            
        elif "raw_response" in result:
            # Raw text - try to parse it
            raw = result["raw_response"]
            
            # Try to find speaker patterns in raw text
            import re
            lines = raw.split('\n')
            turn_num = 0
            seen_texts = set()  # For deduplication
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for various speaker patterns (Speaker1, Speech 1, Host, etc.)
                match = re.match(r'^(Speaker\s*\d+|Speech\s*\d+|Surya|Chandni|Host|Guest|[A-Za-z]+)\s*[:：]\s*(.+)', line, re.IGNORECASE)
                if match:
                    text = match.group(2).strip()
                    
                    # Skip duplicate lines (repetition from LLM)
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)
                    
                    # Normalize to Speaker1/Speaker2
                    if turn_num % 2 == 0:
                        speaker = "Speaker1"
                    else:
                        speaker = "Speaker2"
                    if text:
                        script_lines.append(f"{speaker}: {text}")
                        turn_num += 1
            
            title = "Generated Script"
        else:
            title = "Script"
        
        if not script_lines:
            # Fallback: Show raw response if parsing failed
            if "raw_response" in result:
                # Return raw LLM output for user to manually edit
                return result["raw_response"], "⚠️ Raw output (please format as Speaker1:/Speaker2:)"
            elif "turns" in result:
                # JSON was parsed but no valid turns found
                import json
                return json.dumps(result, ensure_ascii=False, indent=2), "⚠️ JSON output (please format manually)"
            else:
                # No output at all
                return "", "❌ No script generated. Try again."
        
        progress(1.0, desc="Done!")
        script = "\n".join(script_lines)
        return script, f"✅ Generated: {title} ({len(script_lines)} turns)"
        
    except Exception as e:
        return "", f"❌ Error: {str(e)}"


def generate_podcast(
    script: str,
    language: str,
    speaker1_name: str,
    speaker1_emotion: str,
    speaker1_pace: str,
    speaker1_pitch: str,
    speaker1_expressivity: str,
    speaker1_ref: Optional[str],
    speaker2_name: str,
    speaker2_emotion: str,
    speaker2_pace: str,
    speaker2_pitch: str,
    speaker2_expressivity: str,
    speaker2_ref: Optional[str],
    tts_temperature: float,
    tts_top_p: float,
    tts_repetition_penalty: float,
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
    
    # Extract speaker names from labels like "Rohit (male) - Clear, energetic"
    speaker1_name = get_speaker_name_from_label(speaker1_name)
    speaker2_name = get_speaker_name_from_label(speaker2_name)
    
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
        
        # Create per-speaker TTS configs with advanced parameters
        config1 = TTSConfig(
            pace=speaker1_pace,
            pitch=speaker1_pitch,
            expressivity=speaker1_expressivity,
            emotion=speaker1_emotion,
            temperature=tts_temperature,
            top_p=tts_top_p,
            repetition_penalty=tts_repetition_penalty
        )
        config2 = TTSConfig(
            pace=speaker2_pace,
            pitch=speaker2_pitch,
            expressivity=speaker2_expressivity,
            emotion=speaker2_emotion,
            temperature=tts_temperature,
            top_p=tts_top_p,
            repetition_penalty=tts_repetition_penalty
        )
        
        # Generate audio for each turn
        audio_clips = []
        
        for i, (speaker_num, text) in enumerate(turns):
            progress(0.2 + (0.6 * i / len(turns)), desc=f"Generating turn {i+1}/{len(turns)}...")
            
            if speaker_num == 1:
                speaker_name = speaker1_name
                config = config1
                use_clone = use_clone_s1
                ref_audio = speaker1_ref
            else:
                speaker_name = speaker2_name
                config = config2
                use_clone = use_clone_s2
                ref_audio = speaker2_ref
            
            if use_clone and ref_audio:
                # Use XTTS for cloned voice
                audio = xtts.generate(text, ref_audio, language)
            else:
                # Use Indic-ParlerTTS
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
                        
                        # Gemini API key for best quality
                        gemini_key = gr.Textbox(
                            label="🔑 Gemini API Key (recommended for best quality)",
                            placeholder="Paste your Gemini API key here (get free: https://aistudio.google.com/app/apikey)",
                            type="password",
                            info="Without API key, uses local LLM (may produce lower quality)"
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
                                    value=default_speakers[0] if default_speakers else "Rohit (male)",
                                    label="Voice"
                                )
                                speaker1_emotion = gr.Dropdown(
                                    choices=EMOTIONS,
                                    value="conversation",
                                    label="Emotion"
                                )
                                with gr.Row():
                                    speaker1_pace = gr.Dropdown(
                                        choices=["slow", "moderate", "fast"],
                                        value="moderate", label="Pace"
                                    )
                                    speaker1_pitch = gr.Dropdown(
                                        choices=["low", "moderate", "high"],
                                        value="moderate", label="Pitch"
                                    )
                                speaker1_expressivity = gr.Dropdown(
                                    choices=["monotone", "slightly expressive", "very expressive"],
                                    value="slightly expressive", label="Expressivity"
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
                                    value=default_speakers[1] if len(default_speakers) > 1 else "Divya (female)",
                                    label="Voice"
                                )
                                speaker2_emotion = gr.Dropdown(
                                    choices=EMOTIONS,
                                    value="conversation",
                                    label="Emotion"
                                )
                                with gr.Row():
                                    speaker2_pace = gr.Dropdown(
                                        choices=["slow", "moderate", "fast"],
                                        value="moderate", label="Pace"
                                    )
                                    speaker2_pitch = gr.Dropdown(
                                        choices=["low", "moderate", "high"],
                                        value="moderate", label="Pitch"
                                    )
                                speaker2_expressivity = gr.Dropdown(
                                    choices=["monotone", "slightly expressive", "very expressive"],
                                    value="slightly expressive", label="Expressivity"
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
                            
                            gr.Markdown("**TTS Generation** (affects prosody & expressiveness)")
                            with gr.Row():
                                tts_temperature = gr.Slider(
                                    minimum=0.3, maximum=1.2, value=0.7, step=0.1,
                                    label="Temperature",
                                    info="0.6-0.9 conversational, 0.9-1.2 dramatic"
                                )
                                tts_top_p = gr.Slider(
                                    minimum=0.2, maximum=1.0, value=0.9, step=0.1,
                                    label="Top-p",
                                    info="0.6-0.8 natural, 0.8-1.0 expressive"
                                )
                                tts_repetition_penalty = gr.Slider(
                                    minimum=1.0, maximum=2.0, value=1.1, step=0.1,
                                    label="Repetition Penalty",
                                    info="≥1.1 for stable generation"
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
                                add_noise = gr.Checkbox(label="Add background noise", value=False)
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
                
                # Extract text from file (automatic on upload)
                file_upload.change(
                    fn=extract_document_text,
                    inputs=[file_upload],
                    outputs=[content_text, extract_status]
                )
                
                # Also keep extract button for manual use
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
                        num_turns, gemini_key
                    ],
                    outputs=[podcast_script, script_status]
                )
                
                # Generate podcast
                generate_podcast_btn.click(
                    fn=generate_podcast,
                    inputs=[
                        podcast_script, script_language,
                        speaker1_name, speaker1_emotion, 
                        speaker1_pace, speaker1_pitch, speaker1_expressivity,
                        speaker1_ref,
                        speaker2_name, speaker2_emotion,
                        speaker2_pace, speaker2_pitch, speaker2_expressivity,
                        speaker2_ref,
                        tts_temperature, tts_top_p, tts_repetition_penalty,
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
