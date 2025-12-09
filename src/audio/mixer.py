"""
Audio Mixer - Join, crossfade, and process audio clips

Ported from kokoro-podcast-generator with enhancements.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import io


@dataclass
class AudioClip:
    """Audio clip with metadata"""
    audio: np.ndarray
    sample_rate: int
    speaker: str = ""
    text: str = ""


def trim_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    min_silence_samples: int = 100
) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.
    
    Args:
        audio: Audio numpy array
        threshold: Amplitude threshold for silence detection
        min_silence_samples: Minimum samples to consider as silence
        
    Returns:
        Trimmed audio
    """
    # Find first non-silent sample
    start = 0
    for i in range(len(audio)):
        if abs(audio[i]) > threshold:
            start = max(0, i - min_silence_samples)
            break
    
    # Find last non-silent sample
    end = len(audio)
    for i in range(len(audio) - 1, -1, -1):
        if abs(audio[i]) > threshold:
            end = min(len(audio), i + min_silence_samples)
            break
    
    return audio[start:end]


def add_silence(
    audio: np.ndarray,
    duration_ms: int,
    sample_rate: int,
    at_beginning: bool = True
) -> np.ndarray:
    """
    Add silence to audio.
    
    Args:
        audio: Audio numpy array
        duration_ms: Duration of silence in milliseconds
        sample_rate: Sample rate
        at_beginning: Add silence at beginning (True) or end (False)
        
    Returns:
        Audio with silence added
    """
    silence_samples = int((duration_ms / 1000) * sample_rate)
    silence = np.zeros(silence_samples, dtype=audio.dtype)
    
    if at_beginning:
        return np.concatenate([silence, audio])
    else:
        return np.concatenate([audio, silence])


def join_audio(
    audio1: np.ndarray,
    audio2: np.ndarray,
    gap_ms: int = 100,
    sample_rate: int = 24000,
    crossfade: bool = False
) -> np.ndarray:
    """
    Join two audio clips with optional gap or crossfade.
    
    Args:
        audio1: First audio clip
        audio2: Second audio clip
        gap_ms: Gap in milliseconds (negative for overlap)
        sample_rate: Sample rate
        crossfade: Apply crossfade for overlaps
        
    Returns:
        Joined audio
    """
    if gap_ms >= 0:
        # Add silence gap between clips
        gap_samples = int((gap_ms / 1000) * sample_rate)
        silence = np.zeros(gap_samples, dtype=audio1.dtype)
        return np.concatenate([audio1, silence, audio2])
    else:
        # Overlap the clips
        overlap_samples = int((-gap_ms / 1000) * sample_rate)
        overlap_samples = min(overlap_samples, len(audio1), len(audio2))
        
        if crossfade:
            # Linear crossfade
            fade_out = np.linspace(1, 0, overlap_samples)
            fade_in = np.linspace(0, 1, overlap_samples)
            
            # Apply crossfade
            audio1_end = audio1[-overlap_samples:] * fade_out
            audio2_start = audio2[:overlap_samples] * fade_in
            overlap_region = audio1_end + audio2_start
            
            return np.concatenate([
                audio1[:-overlap_samples],
                overlap_region,
                audio2[overlap_samples:]
            ])
        else:
            # Simple overlap (sum)
            result = np.concatenate([audio1, np.zeros(len(audio2) - overlap_samples)])
            result[-len(audio2):] += audio2
            return result


def add_background_noise(
    audio: np.ndarray,
    noise_level: float = 0.002
) -> np.ndarray:
    """
    Add subtle background noise for realism.
    
    Args:
        audio: Audio numpy array
        noise_level: Noise amplitude (0.001-0.01 recommended)
        
    Returns:
        Audio with background noise
    """
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise.astype(audio.dtype)


def normalize_volume(
    audio: np.ndarray,
    target_peak: float = 0.9
) -> np.ndarray:
    """
    Normalize audio volume.
    
    Args:
        audio: Audio numpy array
        target_peak: Target peak amplitude (0-1)
        
    Returns:
        Normalized audio
    """
    current_peak = np.max(np.abs(audio))
    if current_peak > 0:
        scale = target_peak / current_peak
        return audio * scale
    return audio


def mix_background_music(
    speech: np.ndarray,
    music: np.ndarray,
    music_volume: float = 0.1,
    sample_rate: int = 24000,
    fade_in_ms: int = 2000,
    fade_out_ms: int = 2000
) -> np.ndarray:
    """
    Mix background music with speech.
    
    Args:
        speech: Speech audio
        music: Background music
        music_volume: Volume level for music (0-1)
        sample_rate: Sample rate
        fade_in_ms: Fade in duration
        fade_out_ms: Fade out duration
        
    Returns:
        Mixed audio
    """
    # Ensure music is long enough
    if len(music) < len(speech):
        # Loop music
        repeats = (len(speech) // len(music)) + 1
        music = np.tile(music, repeats)
    
    # Trim to match speech length
    music = music[:len(speech)]
    
    # Apply volume
    music = music * music_volume
    
    # Apply fade in
    fade_in_samples = int((fade_in_ms / 1000) * sample_rate)
    if fade_in_samples > 0:
        fade_in = np.linspace(0, 1, fade_in_samples)
        music[:fade_in_samples] *= fade_in
    
    # Apply fade out
    fade_out_samples = int((fade_out_ms / 1000) * sample_rate)
    if fade_out_samples > 0:
        fade_out = np.linspace(1, 0, fade_out_samples)
        music[-fade_out_samples:] *= fade_out
    
    return speech + music


class PodcastMixer:
    """
    Mixes podcast turns into a final audio file.
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        
    def mix_turns(
        self,
        turns: List[AudioClip],
        gap_ms: int = 100,
        add_noise: bool = True,
        noise_level: float = 0.002
    ) -> np.ndarray:
        """
        Mix podcast turns into a single audio file.
        
        Args:
            turns: List of AudioClip objects
            gap_ms: Gap between turns in milliseconds
            add_noise: Add background noise
            noise_level: Noise level
            
        Returns:
            Mixed audio
        """
        if not turns:
            return np.array([])
            
        # Trim silence from each turn
        result = trim_silence(turns[0].audio)
        
        for i in range(1, len(turns)):
            turn_audio = trim_silence(turns[i].audio)
            
            # Use crossfade for negative gaps
            crossfade = gap_ms < 0
            result = join_audio(
                result, turn_audio,
                gap_ms=gap_ms,
                sample_rate=self.sample_rate,
                crossfade=crossfade
            )
        
        # Add intro silence
        result = add_silence(result, 500, self.sample_rate, at_beginning=True)
        
        # Add background noise
        if add_noise:
            result = add_background_noise(result, noise_level)
            
        # Normalize
        result = normalize_volume(result)
        
        return result
    
    def add_intro(
        self,
        podcast_audio: np.ndarray,
        intro_audio: np.ndarray,
        crossfade_ms: int = 2000
    ) -> np.ndarray:
        """
        Add intro music to podcast.
        
        Args:
            podcast_audio: Main podcast audio
            intro_audio: Intro music
            crossfade_ms: Crossfade duration
            
        Returns:
            Audio with intro
        """
        return join_audio(
            intro_audio,
            podcast_audio,
            gap_ms=-crossfade_ms,
            sample_rate=self.sample_rate,
            crossfade=True
        )


if __name__ == "__main__":
    print("Audio mixer module ready.")
    print("Functions: trim_silence, join_audio, add_background_noise, normalize_volume")
