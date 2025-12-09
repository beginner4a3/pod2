"""
Indic-ParlerTTS Wrapper

A comprehensive wrapper for the Indic-ParlerTTS model from AI4Bharat.
Supports 21 Indian languages, 69 speakers, and 12 emotions.

Model: https://huggingface.co/ai4bharat/indic-parler-tts
"""

import torch
import soundfile as sf
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import io


@dataclass
class Speaker:
    """Speaker information"""
    name: str
    language: str
    gender: str  # "male" or "female"
    recommended: bool = False


@dataclass 
class TTSConfig:
    """TTS generation configuration"""
    emotion: str = "neutral"
    pace: str = "moderate"  # slow, moderate, fast
    pitch: str = "moderate"  # low, moderate, high
    expressivity: str = "slightly expressive"
    quality: str = "very high quality"
    background: str = "no background noise"


# All 69 speakers organized by language
SPEAKERS = {
    "hindi": [
        Speaker("Rohit", "hindi", "male", recommended=True),
        Speaker("Divya", "hindi", "female", recommended=True),
        Speaker("Aman", "hindi", "male"),
        Speaker("Rani", "hindi", "female"),
    ],
    "telugu": [
        Speaker("Prakash", "telugu", "male", recommended=True),
        Speaker("Lalitha", "telugu", "female", recommended=True),
        Speaker("Kiran", "telugu", "male"),
    ],
    "tamil": [
        Speaker("Jaya", "tamil", "female", recommended=True),
        Speaker("Kavitha", "tamil", "female"),
    ],
    "malayalam": [
        Speaker("Harish", "malayalam", "male", recommended=True),
        Speaker("Anjali", "malayalam", "female", recommended=True),
        Speaker("Anju", "malayalam", "female"),
    ],
    "kannada": [
        Speaker("Suresh", "kannada", "male", recommended=True),
        Speaker("Anu", "kannada", "female", recommended=True),
        Speaker("Chetan", "kannada", "male"),
        Speaker("Vidya", "kannada", "female"),
    ],
    "bengali": [
        Speaker("Arjun", "bengali", "male", recommended=True),
        Speaker("Aditi", "bengali", "female", recommended=True),
        Speaker("Tapan", "bengali", "male"),
        Speaker("Rashmi", "bengali", "female"),
        Speaker("Arnav", "bengali", "male"),
        Speaker("Riya", "bengali", "female"),
    ],
    "marathi": [
        Speaker("Sanjay", "marathi", "male", recommended=True),
        Speaker("Sunita", "marathi", "female", recommended=True),
        Speaker("Nikhil", "marathi", "male"),
        Speaker("Radha", "marathi", "female"),
        Speaker("Varun", "marathi", "male"),
        Speaker("Isha", "marathi", "female"),
    ],
    "gujarati": [
        Speaker("Yash", "gujarati", "male", recommended=True),
        Speaker("Neha", "gujarati", "female", recommended=True),
    ],
    "english": [
        Speaker("Thoma", "english", "male", recommended=True),
        Speaker("Mary", "english", "female", recommended=True),
        Speaker("Swapna", "english", "female"),
        Speaker("Dinesh", "english", "male"),
        Speaker("Meera", "english", "female"),
        Speaker("Jatin", "english", "male"),
        Speaker("Aakash", "english", "male"),
        Speaker("Sneha", "english", "female"),
        Speaker("Kabir", "english", "male"),
        Speaker("Tisha", "english", "female"),
        Speaker("Priya", "english", "female"),
        Speaker("Tarun", "english", "male"),
        Speaker("Gauri", "english", "female"),
        Speaker("Nisha", "english", "female"),
        Speaker("Raghav", "english", "male"),
        Speaker("Kavya", "english", "female"),
        Speaker("Ravi", "english", "male"),
        Speaker("Vikas", "english", "male"),
        Speaker("Riya", "english", "female"),
    ],
    "assamese": [
        Speaker("Amit", "assamese", "male", recommended=True),
        Speaker("Sita", "assamese", "female", recommended=True),
        Speaker("Poonam", "assamese", "female"),
        Speaker("Rakesh", "assamese", "male"),
    ],
    "bodo": [
        Speaker("Bikram", "bodo", "male", recommended=True),
        Speaker("Maya", "bodo", "female", recommended=True),
        Speaker("Kalpana", "bodo", "female"),
    ],
    "dogri": [
        Speaker("Karan", "dogri", "male", recommended=True),
    ],
    "odia": [
        Speaker("Manas", "odia", "male", recommended=True),
        Speaker("Debjani", "odia", "female", recommended=True),
    ],
    "punjabi": [
        Speaker("Divjot", "punjabi", "male", recommended=True),
        Speaker("Gurpreet", "punjabi", "female", recommended=True),
    ],
    "nepali": [
        Speaker("Amrita", "nepali", "female", recommended=True),
    ],
    "sanskrit": [
        Speaker("Aryan", "sanskrit", "male", recommended=True),
    ],
    "manipuri": [
        Speaker("Laishram", "manipuri", "male", recommended=True),
        Speaker("Ranjit", "manipuri", "male", recommended=True),
    ],
}

# Supported emotions
EMOTIONS = [
    "neutral",
    "happy", 
    "sad",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "command",
    "narration",
    "conversation",
    "news",
    "proper_noun",
]

# Emotion to description mapping
EMOTION_DESCRIPTIONS = {
    "neutral": "neutral tone",
    "happy": "happy, cheerful tone full of energy",
    "sad": "sad, melancholic tone",
    "anger": "angry, intense tone",
    "fear": "fearful, anxious tone",
    "surprise": "surprised, excited tone",
    "disgust": "disgusted tone",
    "command": "commanding, authoritative tone",
    "narration": "narrative, storytelling tone",
    "conversation": "conversational, friendly tone",
    "news": "professional news anchor tone",
    "proper_noun": "clear pronunciation for proper nouns",
}


class IndicParlerTTS:
    """
    Wrapper for Indic-ParlerTTS model.
    
    Model: https://huggingface.co/ai4bharat/indic-parler-tts
    
    The model should be pre-downloaded. First-time setup requires:
    1. Set HF_TOKEN environment variable
    2. Run: huggingface-cli download ai4bharat/indic-parler-tts
    
    After download, the model runs locally without any token.
    
    Usage:
        tts = IndicParlerTTS()
        audio = tts.generate("नमस्ते, आज हम AI के बारे में बात करेंगे", speaker="Rohit", emotion="happy")
        tts.save(audio, "output.wav")
    """
    
    MODEL_ID = "ai4bharat/indic-parler-tts"
    MODEL_URL = "https://huggingface.co/ai4bharat/indic-parler-tts"
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the TTS model.
        
        Args:
            device: Device to use ("cuda", "cpu", or None for auto-detect)
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        self.sample_rate = 24000  # Default sample rate
        
    def load(self):
        """
        Load the pre-downloaded model and tokenizers.
        
        The model should already be cached in HuggingFace cache directory.
        """
        if self.model is not None:
            return  # Already loaded
            
        print(f"Loading Indic-ParlerTTS model on {self.device}...")
        print(f"Model: {self.MODEL_URL}")
        
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        # Load model from cache (no token needed - already downloaded)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.MODEL_ID
        ).to(self.device)
        
        # Load tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        
        self.sample_rate = self.model.config.sampling_rate
        print(f"✅ Model loaded successfully! Sample rate: {self.sample_rate}")
        
    def _build_description(
        self,
        speaker: str,
        config: Optional[TTSConfig] = None
    ) -> str:
        """
        Build a description string for the model.
        
        Args:
            speaker: Speaker name (e.g., "Rohit", "Divya")
            config: TTS configuration
            
        Returns:
            Description string for the model
        """
        if config is None:
            config = TTSConfig()
            
        emotion_desc = EMOTION_DESCRIPTIONS.get(config.emotion, config.emotion)
        
        # Build description following the model's expected format
        description = (
            f"{speaker}'s voice is {emotion_desc} with a {config.pace} pace. "
            f"The recording is of {config.quality}, with the speaker's voice "
            f"sounding clear and very close up with {config.background}."
        )
        
        return description
    
    def generate(
        self,
        text: str,
        speaker: str = "Rohit",
        emotion: str = "neutral",
        pace: str = "moderate",
        config: Optional[TTSConfig] = None
    ) -> np.ndarray:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech (any supported language)
            speaker: Speaker name (e.g., "Rohit", "Divya", "Prakash")
            emotion: Emotion (e.g., "happy", "sad", "neutral")
            pace: Speaking pace ("slow", "moderate", "fast")
            config: Full TTS configuration (overrides emotion/pace if provided)
            
        Returns:
            Audio as numpy array
        """
        self.load()  # Ensure model is loaded
        
        # Build config
        if config is None:
            config = TTSConfig(emotion=emotion, pace=pace)
            
        # Build description
        description = self._build_description(speaker, config)
        
        # Tokenize
        description_input_ids = self.description_tokenizer(
            description, return_tensors="pt"
        ).to(self.device)
        
        prompt_input_ids = self.tokenizer(
            text, return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generation = self.model.generate(
                input_ids=description_input_ids.input_ids,
                attention_mask=description_input_ids.attention_mask,
                prompt_input_ids=prompt_input_ids.input_ids,
                prompt_attention_mask=prompt_input_ids.attention_mask
            )
            
        audio = generation.cpu().numpy().squeeze()
        return audio
    
    def generate_podcast_turn(
        self,
        text: str,
        speaker: str,
        language: str = "hindi",
        emotion: str = "conversation"
    ) -> np.ndarray:
        """
        Generate a single podcast turn with appropriate settings.
        
        Args:
            text: Text content
            speaker: Speaker name
            language: Target language
            emotion: Emotion for this turn
            
        Returns:
            Audio as numpy array
        """
        config = TTSConfig(
            emotion=emotion,
            pace="moderate",
            expressivity="slightly expressive",
            quality="very high quality",
            background="no background noise"
        )
        
        return self.generate(text, speaker=speaker, config=config)
    
    def save(self, audio: np.ndarray, path: str):
        """
        Save audio to file.
        
        Args:
            audio: Audio numpy array
            path: Output file path (.wav)
        """
        sf.write(path, audio, self.sample_rate)
        
    def to_bytes(self, audio: np.ndarray) -> bytes:
        """
        Convert audio to bytes (WAV format).
        
        Args:
            audio: Audio numpy array
            
        Returns:
            WAV file as bytes
        """
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()
    
    @staticmethod
    def get_speakers(language: Optional[str] = None) -> List[Speaker]:
        """
        Get list of available speakers.
        
        Args:
            language: Filter by language (optional)
            
        Returns:
            List of Speaker objects
        """
        if language:
            return SPEAKERS.get(language.lower(), [])
        
        all_speakers = []
        for lang_speakers in SPEAKERS.values():
            all_speakers.extend(lang_speakers)
        return all_speakers
    
    @staticmethod
    def get_recommended_speakers(language: str) -> Tuple[str, str]:
        """
        Get recommended speaker pair for a language.
        
        Args:
            language: Language code
            
        Returns:
            Tuple of (speaker1, speaker2) names
        """
        speakers = SPEAKERS.get(language.lower(), SPEAKERS["hindi"])
        recommended = [s for s in speakers if s.recommended]
        
        if len(recommended) >= 2:
            return recommended[0].name, recommended[1].name
        elif len(recommended) == 1:
            return recommended[0].name, speakers[0].name if speakers else "Rohit"
        else:
            return "Rohit", "Divya"  # Default fallback
    
    @staticmethod
    def get_languages() -> List[str]:
        """Get list of supported languages."""
        return list(SPEAKERS.keys())
    
    @staticmethod
    def get_emotions() -> List[str]:
        """Get list of supported emotions."""
        return EMOTIONS


# Convenience function
def create_tts(device: Optional[str] = None) -> IndicParlerTTS:
    """Create and return an IndicParlerTTS instance."""
    return IndicParlerTTS(device=device)


if __name__ == "__main__":
    # Quick test
    print("Testing Indic-ParlerTTS...")
    print(f"Supported languages: {IndicParlerTTS.get_languages()}")
    print(f"Hindi speakers: {[s.name for s in IndicParlerTTS.get_speakers('hindi')]}")
    print(f"Emotions: {IndicParlerTTS.get_emotions()}")
