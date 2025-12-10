"""
Google Gemini API for Podcast Script Generation

This module provides high-quality script generation using Google's Gemini API.
It produces much better scripts than small local LLMs without hallucinations.

Usage:
    from src.llm.gemini_api import GeminiScriptGenerator
    
    generator = GeminiScriptGenerator(api_key="your_gemini_api_key")
    script = generator.generate_script(topic="Dual Nature of Light", language="hindi")
"""

import os
import json
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class GeminiConfig:
    """Configuration for Gemini API"""
    api_key: str = ""
    model: str = "gemini-1.5-flash"  # Fast and affordable
    temperature: float = 0.7
    max_tokens: int = 4096


class GeminiScriptGenerator:
    """
    Generate podcast scripts using Google Gemini API.
    
    This produces much higher quality scripts than small local LLMs,
    with proper Hindi/English code-mixing and no hallucinations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini API client.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = "gemini-1.5-flash"
        self._client = None
        
    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                )
        return self._client
    
    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        return bool(self.api_key)
    
    def _get_system_prompt(self, language: str) -> str:
        """Get system prompt optimized for Indic Parler-TTS."""
        
        lang_config = {
            "hindi": {"script": "Devanagari", "example": "नमस्ते दोस्तों"},
            "telugu": {"script": "Telugu", "example": "నమస్కారం"},
            "tamil": {"script": "Tamil", "example": "வணக்கம்"},
            "malayalam": {"script": "Malayalam", "example": "നമസ്കാരം"},
            "kannada": {"script": "Kannada", "example": "ನಮಸ್ಕಾರ"},
            "bengali": {"script": "Bengali", "example": "নমস্কার"},
            "marathi": {"script": "Devanagari", "example": "नमस्कार"},
            "gujarati": {"script": "Gujarati", "example": "નમસ્તે"},
        }
        
        config = lang_config.get(language.lower(), lang_config["hindi"])
        
        return f"""You are an expert podcast script writer optimized for Indic Parler-TTS.

Your task is to generate a two-speaker podcast script that will be converted to speech.

### CRITICAL RULES FOR TTS-READY SCRIPTS:

**RULE 1 - LANGUAGE MIXING ({language} + English ONLY):**
- Write {language} words in {config['script']} script ONLY
- Keep English technical terms (photon, electron, energy, wave, particle, theory) in ENGLISH/Latin script
- ✅ CORRECT: "Light की dual nature होती है, यानी wave और particle दोनों।"
- ❌ WRONG: "Light ki dual nature hoti hai" (Romanized Hindi)
- ❌ WRONG: Random French/Spanish/Chinese/Armenian/Bengali characters

**RULE 2 - PUNCTUATION FOR NATURAL SPEECH:**
- Add commas (,) every 5-6 words for natural breathing pauses
- ✅ CORRECT: "सबसे पहले, यह समझते हैं कि, Light की nature कैसी है।"
- ❌ WRONG: "सबसे पहले यह समझते हैं कि Light की nature कैसी है।"

**RULE 3 - FACTUAL ACCURACY:**
- Write scientifically correct content
- Do NOT hallucinate or make up facts
- Do NOT confuse words (e.g., "packets" ≠ "Pakistan", "matter" ≠ "marta/kills")
- Use correct vocabulary: "माध्यम" (medium), NOT "मंच" (stage)

**RULE 4 - FORMAT:**
- Use ONLY "Speaker1:" and "Speaker2:" prefixes
- Speaker1 = Host/Teacher (explains concepts)
- Speaker2 = Co-host/Student (asks questions, reacts)
- End with proper conclusion (धन्यवाद दोस्तों!)
- NO notes/comments after the script
- Output ONLY the script, nothing else

**RULE 5 - KEEP TECHNICAL TERMS IN ENGLISH:**
These words should ALWAYS be in English: 
Physics, Light, Wave, Particle, Photon, Electron, Energy, Theory, 
Experiment, Quantum, Nobel Prize, Dual Nature, Photoelectric Effect,
Medium, Prism, Rainbow, Camera, Sensor, etc."""
    
    def generate_script(
        self,
        topic: str,
        language: str = "hindi",
        num_turns: int = 15,
        content: Optional[str] = None
    ) -> str:
        """
        Generate a podcast script using Gemini API.
        
        Args:
            topic: Topic for the podcast
            language: Target language (hindi, telugu, tamil, etc.)
            num_turns: Number of dialogue turns
            content: Optional content/context for the script
            
        Returns:
            Generated script with Speaker1/Speaker2 format
        """
        if not self.is_available():
            raise ValueError(
                "Gemini API key not set. "
                "Set GEMINI_API_KEY environment variable or pass api_key to constructor."
            )
        
        client = self._get_client()
        
        system_prompt = self._get_system_prompt(language)
        
        user_prompt = f"""Write a {num_turns}-turn educational podcast script about:

**Topic:** {topic}

"""
        if content:
            user_prompt += f"""**Reference Content:**
{content[:3000]}

Use this content to create an accurate, informative podcast script.

"""
        
        user_prompt += f"""**Requirements:**
1. Language: {language} (native script) mixed with English technical terms
2. Format: "Speaker1:" and "Speaker2:" alternating
3. Add commas every 5-6 words for natural pauses
4. End with "धन्यवाद दोस्तों!" or equivalent

**START WRITING THE {num_turns}-TURN SCRIPT NOW:**"""
        
        try:
            response = client.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 4096,
                }
            )
            
            script = response.text.strip()
            
            # Clean up the script
            script = self._clean_script(script)
            
            return script
            
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    def _clean_script(self, script: str) -> str:
        """Clean and validate the generated script."""
        lines = []
        
        for line in script.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Keep only Speaker1/Speaker2 lines
            if line.startswith("Speaker1:") or line.startswith("Speaker2:"):
                lines.append(line)
            elif ":" in line:
                # Try to fix other speaker formats
                parts = line.split(":", 1)
                speaker_part = parts[0].strip().lower()
                if "speaker" in speaker_part or "1" in speaker_part:
                    lines.append(f"Speaker1:{parts[1]}")
                elif "2" in speaker_part:
                    lines.append(f"Speaker2:{parts[1]}")
        
        return '\n'.join(lines)


def is_gemini_available() -> bool:
    """Check if Gemini API is available."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return bool(api_key)


def generate_script_with_gemini(
    topic: str,
    language: str = "hindi",
    num_turns: int = 15,
    content: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Convenience function to generate script with Gemini.
    
    Args:
        topic: Topic for the podcast
        language: Target language
        num_turns: Number of dialogue turns
        content: Optional reference content
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        Generated script
    """
    generator = GeminiScriptGenerator(api_key=api_key)
    return generator.generate_script(topic, language, num_turns, content)
