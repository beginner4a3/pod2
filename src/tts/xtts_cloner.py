"""
XTTS Voice Cloner - Optional voice cloning using Coqui XTTS v2

This module provides optional voice cloning functionality.
If no reference voice is provided, the system falls back to Indic-ParlerTTS.

Model: https://huggingface.co/coqui/XTTS-v2
Requirements: ~6 second reference audio, 4GB+ VRAM recommended
"""

import os
import torch
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class XTTSCloner:
    """
    Voice cloning using Coqui XTTS v2.
    
    This is optional and supplements (not replaces) Indic-ParlerTTS.
    Only used when a reference voice audio is uploaded.
    
    Usage:
        cloner = XTTSCloner()
        audio = cloner.generate("Hello world", reference_audio="voice.wav")
    """
    
    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize XTTS cloner.
        
        Args:
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sample_rate = 24000
        self._available = None
        
    def is_available(self) -> bool:
        """Check if XTTS is available (TTS library installed)."""
        if self._available is None:
            try:
                from TTS.api import TTS
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def load(self):
        """Load the XTTS v2 model."""
        if not self.is_available():
            raise ImportError(
                "coqui-tts library is required for voice cloning. "
                "Install with: pip install coqui-tts>=0.25.0"
            )
            
        if self.model is not None:
            return  # Already loaded
            
        print(f"Loading XTTS v2 on {self.device}...")
        
        from TTS.api import TTS
        
        self.model = TTS(self.MODEL_NAME).to(self.device)
        self.sample_rate = self.model.synthesizer.output_sample_rate
        
        print(f"✅ XTTS v2 loaded! Sample rate: {self.sample_rate}")
    
    def generate(
        self,
        text: str,
        reference_audio: str,
        language: str = "en"
    ) -> np.ndarray:
        """
        Generate speech with cloned voice.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file (~6s recommended)
            language: Target language code (en, hi, ta, te, etc.)
            
        Returns:
            Audio as numpy array
        """
        self.load()
        
        if not Path(reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        
        # Map common language names to XTTS codes
        lang_map = {
            "hindi": "hi", "english": "en", "tamil": "ta",
            "telugu": "te", "bengali": "bn", "marathi": "mr",
            "gujarati": "gu", "kannada": "kn", "malayalam": "ml",
            "punjabi": "pa", "odia": "or", "assamese": "as",
        }
        lang_code = lang_map.get(language.lower(), language[:2].lower())
        
        # Generate with voice cloning
        audio = self.model.tts(
            text=text,
            speaker_wav=reference_audio,
            language=lang_code
        )
        
        return np.array(audio, dtype=np.float32)
    
    def clone_and_save(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
        language: str = "en"
    ):
        """
        Generate and save cloned voice audio.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio
            output_path: Output file path
            language: Target language
        """
        audio = self.generate(text, reference_audio, language)
        
        import soundfile as sf
        sf.write(output_path, audio, self.sample_rate)
        
        return output_path


def create_cloner(device: Optional[str] = None) -> XTTSCloner:
    """Create an XTTSCloner instance."""
    return XTTSCloner(device=device)


def is_voice_cloning_available() -> bool:
    """Check if voice cloning is available."""
    try:
        from TTS.api import TTS
        return True
    except (ImportError, ValueError, Exception):
        # ImportError: TTS not installed
        # ValueError: numpy binary incompatibility on some Python versions
        return False


if __name__ == "__main__":
    print("XTTS Voice Cloner Module")
    print(f"Voice cloning available: {is_voice_cloning_available()}")
    if is_voice_cloning_available():
        print("Usage:")
        print("  cloner = XTTSCloner()")
        print("  audio = cloner.generate('Hello', reference_audio='voice.wav')")
    else:
        print("Install coqui-tts for voice cloning: pip install coqui-tts>=0.25.0")
