"""
Audio Format Utilities - WAV and MP3 export
"""

import numpy as np
import io
import struct
from typing import Optional


def audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Convert numpy audio to WAV bytes.
    
    Args:
        audio: Audio as numpy array (float32 or int16)
        sample_rate: Sample rate in Hz
        
    Returns:
        WAV file as bytes
    """
    # Convert to 16-bit PCM if needed
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        # Clip and scale to 16-bit
        audio = np.clip(audio, -1, 1)
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    
    # WAV header
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(audio) * (bits_per_sample // 8)
    
    buffer = io.BytesIO()
    
    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))
    buffer.write(b'WAVE')
    
    # fmt chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))   # Audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    
    # data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(audio.tobytes())
    
    buffer.seek(0)
    return buffer.read()


def save_wav(audio: np.ndarray, sample_rate: int, path: str):
    """
    Save audio to WAV file.
    
    Args:
        audio: Audio numpy array
        sample_rate: Sample rate
        path: Output file path
    """
    wav_bytes = audio_to_wav(audio, sample_rate)
    with open(path, 'wb') as f:
        f.write(wav_bytes)


def audio_to_mp3(
    audio: np.ndarray,
    sample_rate: int,
    bitrate: int = 128
) -> bytes:
    """
    Convert numpy audio to MP3 bytes.
    
    Requires pydub and ffmpeg.
    
    Args:
        audio: Audio numpy array
        sample_rate: Sample rate
        bitrate: MP3 bitrate in kbps
        
    Returns:
        MP3 file as bytes
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is required for MP3 export. Install with: pip install pydub")
    
    # Convert to 16-bit PCM
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = np.clip(audio, -1, 1)
        audio = (audio * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        data=audio.tobytes(),
        sample_width=2,  # 16-bit = 2 bytes
        frame_rate=sample_rate,
        channels=1
    )
    
    # Export to MP3
    buffer = io.BytesIO()
    audio_segment.export(buffer, format='mp3', bitrate=f'{bitrate}k')
    buffer.seek(0)
    return buffer.read()


def save_mp3(
    audio: np.ndarray,
    sample_rate: int,
    path: str,
    bitrate: int = 128
):
    """
    Save audio to MP3 file.
    
    Args:
        audio: Audio numpy array
        sample_rate: Sample rate
        path: Output file path
        bitrate: MP3 bitrate in kbps
    """
    mp3_bytes = audio_to_mp3(audio, sample_rate, bitrate)
    with open(path, 'wb') as f:
        f.write(mp3_bytes)


def get_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Get audio duration in seconds.
    
    Args:
        audio: Audio numpy array
        sample_rate: Sample rate
        
    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate


if __name__ == "__main__":
    print("Audio format utilities ready.")
    print("Functions: audio_to_wav, audio_to_mp3, save_wav, save_mp3")
