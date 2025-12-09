"""
Unified Podcast Generator - FastAPI Endpoints

Provides REST API for programmatic podcast generation.

Endpoints:
- POST /upload - Upload document (PDF/DOCX/TXT)
- POST /generate-script - Generate podcast script from content
- POST /generate-podcast - Generate podcast audio
- GET /speakers/{language} - List speakers for a language
- GET /languages - List supported languages
- GET /emotions - List supported emotions
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Initialize FastAPI
app = FastAPI(
    title="Unified Podcast Generator API",
    description="Generate podcasts with Indic-ParlerTTS + optional voice cloning",
    version="1.0.0"
)

# ============================================================================
# Request/Response Models
# ============================================================================

class ScriptRequest(BaseModel):
    """Request model for script generation."""
    content: str = Field(..., description="Document content or topic")
    language: str = Field(default="hindi", description="Target language")
    style: str = Field(default="conversational", description="Script style")
    num_turns: int = Field(default=15, description="Number of conversation turns")


class ScriptResponse(BaseModel):
    """Response model for script generation."""
    script: str
    title: Optional[str] = None
    language: str
    turn_count: int


class SpeakerSettings(BaseModel):
    """Settings for a single speaker."""
    name: str = Field(default="Rohit", description="Speaker name")
    emotion: str = Field(default="conversation", description="Emotion")
    reference_audio_id: Optional[str] = Field(None, description="ID of uploaded reference audio for cloning")


class TTSSettings(BaseModel):
    """TTS configuration."""
    pace: str = Field(default="moderate", description="slow/moderate/fast")
    pitch: str = Field(default="moderate", description="low/moderate/high")
    expressivity: str = Field(default="slightly expressive", description="Expressivity level")


class MixerSettings(BaseModel):
    """Audio mixer configuration."""
    gap_ms: int = Field(default=100, description="Gap between turns in ms")
    crossfade: bool = Field(default=False, description="Enable crossfade")
    intro_silence_ms: int = Field(default=500, description="Intro silence in ms")
    outro_silence_ms: int = Field(default=500, description="Outro silence in ms")
    add_noise: bool = Field(default=True, description="Add background noise")
    noise_level: float = Field(default=0.002, description="Noise level 0.001-0.01")
    normalize: bool = Field(default=True, description="Normalize volume")


class PodcastRequest(BaseModel):
    """Request model for podcast generation."""
    script: str = Field(..., description="Podcast script")
    language: str = Field(default="hindi", description="Language")
    speaker1: SpeakerSettings = Field(default_factory=SpeakerSettings)
    speaker2: SpeakerSettings = Field(default_factory=lambda: SpeakerSettings(name="Divya"))
    tts_settings: TTSSettings = Field(default_factory=TTSSettings)
    mixer_settings: MixerSettings = Field(default_factory=MixerSettings)


class UploadResponse(BaseModel):
    """Response for file upload."""
    file_id: str
    file_name: str
    file_type: str
    word_count: int
    text_preview: str


# ============================================================================
# Storage for uploaded files (in-memory for simplicity)
# ============================================================================

uploaded_files = {}  # file_id -> {"path": path, "text": text, ...}
reference_audios = {}  # audio_id -> file_path


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root - health check."""
    return {"status": "ok", "message": "Unified Podcast Generator API"}


@app.get("/languages")
async def list_languages():
    """List supported languages."""
    from src.tts.indic_parler import IndicParlerTTS
    return {"languages": IndicParlerTTS.get_languages()}


@app.get("/emotions")
async def list_emotions():
    """List supported emotions."""
    from src.tts.indic_parler import IndicParlerTTS
    return {"emotions": IndicParlerTTS.get_emotions()}


@app.get("/speakers/{language}")
async def list_speakers(language: str):
    """List speakers for a language."""
    from src.tts.indic_parler import IndicParlerTTS
    
    speakers = IndicParlerTTS.get_speakers(language)
    if not speakers:
        raise HTTPException(status_code=404, detail=f"Language not found: {language}")
    
    return {
        "language": language,
        "speakers": [
            {"name": s.name, "gender": s.gender, "recommended": s.recommended}
            for s in speakers
        ]
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT) for processing.
    
    Returns file_id to use in subsequent requests.
    """
    from src.script.document_parser import parse_document
    
    # Check file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {suffix}. Supported: .pdf, .docx, .txt"
        )
    
    # Save to temp file
    file_id = str(uuid.uuid4())
    temp_path = Path(tempfile.gettempdir()) / f"{file_id}{suffix}"
    
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Parse document
        result = parse_document(str(temp_path))
        
        # Store
        uploaded_files[file_id] = {
            "path": str(temp_path),
            "text": result["text"],
            "file_name": file.filename,
            "file_type": result["file_type"],
            "word_count": result["word_count"],
        }
        
        return UploadResponse(
            file_id=file_id,
            file_name=file.filename,
            file_type=result["file_type"],
            word_count=result["word_count"],
            text_preview=result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload-reference-audio")
async def upload_reference_audio(file: UploadFile = File(...)):
    """
    Upload a reference audio file for voice cloning.
    
    Recommended: ~6 seconds of clear speech.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".wav", ".mp3", ".flac", ".ogg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}. Supported: .wav, .mp3, .flac, .ogg"
        )
    
    audio_id = str(uuid.uuid4())
    temp_path = Path(tempfile.gettempdir()) / f"ref_{audio_id}{suffix}"
    
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    reference_audios[audio_id] = str(temp_path)
    
    return {
        "audio_id": audio_id,
        "file_name": file.filename,
        "message": "Reference audio uploaded. Use audio_id in speaker settings for voice cloning."
    }


@app.post("/generate-script", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    """
    Generate a podcast script from content using local LLM.
    
    Requires a GGUF model to be configured.
    """
    from src.llm.llama_local import LlamaLocalLLM
    
    try:
        llm = LlamaLocalLLM()
        # Note: This will fail if no model is loaded
        # In production, check model availability first
        result = llm.generate_podcast_script(
            topic=request.content,
            language=request.language,
            num_turns=request.num_turns,
            style=request.style
        )
        
        if "raw_response" in result:
            # Fallback: return raw response as script
            return ScriptResponse(
                script=result["raw_response"],
                language=request.language,
                turn_count=request.num_turns
            )
        
        # Parse structured response
        turns = result.get("turns", [])
        script_lines = []
        for turn in turns:
            speaker = turn.get("speaker", "Speaker1")
            text = turn.get("text", "")
            script_lines.append(f"{speaker}: {text}")
        
        return ScriptResponse(
            script="\n".join(script_lines),
            title=result.get("title"),
            language=request.language,
            turn_count=len(turns)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Script generation failed: {e}")


@app.post("/generate-podcast")
async def generate_podcast(request: PodcastRequest):
    """
    Generate a podcast from script.
    
    Uses Indic-ParlerTTS by default.
    Uses XTTS v2 for voice cloning if reference_audio_id is provided.
    """
    from src.tts.indic_parler import IndicParlerTTS, TTSConfig
    from src.audio.mixer import PodcastMixer, AudioClip
    from src.tts.xtts_cloner import XTTSCloner, is_voice_cloning_available
    
    try:
        # Parse script
        lines = request.script.strip().split("\n")
        turns = []
        
        for line in lines:
            line = line.strip()
            if not line or ":" not in line:
                continue
            speaker_part, text = line.split(":", 1)
            speaker_part = speaker_part.strip().lower()
            text = text.strip()
            
            if "speaker1" in speaker_part or request.speaker1.name.lower() in speaker_part:
                turns.append({"speaker": 1, "text": text})
            elif "speaker2" in speaker_part or request.speaker2.name.lower() in speaker_part:
                turns.append({"speaker": 2, "text": text})
            else:
                # Alternate
                speaker_num = 1 if len(turns) % 2 == 0 else 2
                turns.append({"speaker": speaker_num, "text": text})
        
        if not turns:
            raise HTTPException(status_code=400, detail="Could not parse script")
        
        # Initialize TTS engines
        indic_tts = IndicParlerTTS()
        xtts_cloner = None
        
        # Check voice cloning
        use_cloning_s1 = (
            request.speaker1.reference_audio_id and 
            request.speaker1.reference_audio_id in reference_audios and
            is_voice_cloning_available()
        )
        use_cloning_s2 = (
            request.speaker2.reference_audio_id and 
            request.speaker2.reference_audio_id in reference_audios and
            is_voice_cloning_available()
        )
        
        if use_cloning_s1 or use_cloning_s2:
            xtts_cloner = XTTSCloner()
        
        # Generate audio for each turn
        audio_clips = []
        tts_config = TTSConfig(
            emotion=request.speaker1.emotion,
            pace=request.tts_settings.pace,
            pitch=request.tts_settings.pitch,
            expressivity=request.tts_settings.expressivity
        )
        
        for turn in turns:
            speaker_num = turn["speaker"]
            text = turn["text"]
            
            if speaker_num == 1:
                speaker = request.speaker1
                use_cloning = use_cloning_s1
            else:
                speaker = request.speaker2
                use_cloning = use_cloning_s2
            
            if use_cloning and xtts_cloner:
                # Use XTTS for cloned voice
                ref_path = reference_audios[speaker.reference_audio_id]
                audio = xtts_cloner.generate(text, ref_path, request.language)
            else:
                # Use Indic-ParlerTTS
                audio = indic_tts.generate(
                    text, 
                    speaker=speaker.name, 
                    emotion=speaker.emotion,
                    config=tts_config
                )
            
            audio_clips.append(AudioClip(
                audio=audio,
                sample_rate=indic_tts.sample_rate,
                speaker=speaker.name,
                text=text
            ))
        
        # Mix audio
        mixer = PodcastMixer(sample_rate=indic_tts.sample_rate)
        final_audio = mixer.mix_turns(
            audio_clips,
            gap_ms=request.mixer_settings.gap_ms,
            add_noise=request.mixer_settings.add_noise,
            noise_level=request.mixer_settings.noise_level
        )
        
        # Save to temp file
        output_id = str(uuid.uuid4())
        output_path = Path(tempfile.gettempdir()) / f"podcast_{output_id}.wav"
        
        import soundfile as sf
        sf.write(str(output_path), final_audio, indic_tts.sample_rate)
        
        return FileResponse(
            str(output_path),
            media_type="audio/wav",
            filename=f"podcast_{output_id}.wav"
        )
        
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Podcast generation failed: {e}\n{traceback.format_exc()}")


# ============================================================================
# Run with: uvicorn src.api:app --reload
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
