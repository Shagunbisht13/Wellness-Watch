"""
Module 4 — Audio Processing Module
==================================
Purpose:
    Process the raw speech signal from the microphone into a clean waveform
    suitable for STT, feature extraction, and speech-emotion models.

Technologies:
    sounddevice (capture upstream), Librosa, NumPy, SciPy.

Steps:
    record → chunk → convert to waveform → noise reduction → normalize amplitude

Output:
    Clean audio signal (float32 mono waveform). Pitch/tone/energy features
    are extracted later by Module 7.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
from scipy.io.wavfile import write as wav_write

AudioInput = Union[np.ndarray, str]


def _to_float_mono(audio: np.ndarray) -> np.ndarray:
    """Convert int/float audio arrays to float32 mono in roughly [-1, 1]."""
    y = np.asarray(audio)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if np.issubdtype(np.asarray(audio).dtype, np.integer):
        max_abs = float(np.iinfo(np.asarray(audio).dtype).max)
        y = y / max_abs
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.5:
        y = y / peak
    return y


def _reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Lightweight noise reduction: high-pass filter + soft spectral floor.
    Avoids an extra dependency while cleaning typical mic hiss.
    """
    try:
        import librosa
        from scipy.signal import butter, filtfilt

        # High-pass ~80 Hz to drop rumble
        b, a = butter(N=4, Wn=80 / (sr / 2), btype="high")
        y = filtfilt(b, a, y).astype(np.float32)

        # Soft noise gate using a short noise floor estimate from quiet frames
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
        noise_floor = float(np.percentile(rms, 20))
        if noise_floor > 0:
            # Attenuate frames near the noise floor
            frame_len, hop = 512, 128
            out = y.copy()
            for i, energy in enumerate(rms):
                if energy < noise_floor * 1.5:
                    start = i * hop
                    end = min(len(out), start + frame_len)
                    out[start:end] *= 0.35
            y = out
        return y.astype(np.float32)
    except Exception as exc:
        print(f"[AudioProcessing] Noise reduction fallback: {exc}")
        return y


def _normalize_amplitude(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak-normalize the waveform."""
    if y.size == 0:
        return y
    max_abs = float(np.max(np.abs(y)))
    if max_abs < 1e-8:
        return y
    return (y / max_abs * peak).astype(np.float32)


def process_audio_chunk(
    audio: AudioInput,
    sample_rate: int = 16000,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Convert a raw audio chunk (or WAV path) into a clean float32 waveform.

    Parameters
    ----------
    audio : np.ndarray | str
        Raw chunk from Module 2, or path to a WAV file.
    sample_rate : int
        Sample rate of the chunk (also used when writing output).
    output_path : str | None
        If set, write a 16-bit PCM WAV of the cleaned signal.

    Returns
    -------
    np.ndarray
        Clean mono float32 waveform.
    """
    if isinstance(audio, str):
        try:
            import librosa
            y, sr = librosa.load(audio, sr=sample_rate, mono=True)
            sample_rate = sr
        except Exception:
            from scipy.io.wavfile import read as wav_read
            sr, data = wav_read(audio)
            y = _to_float_mono(data)
            if sr != sample_rate:
                try:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                except Exception:
                    sample_rate = sr
    else:
        y = _to_float_mono(audio)

    y = _reduce_noise(y, sample_rate)
    y = _normalize_amplitude(y)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        pcm = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
        wav_write(output_path, sample_rate, pcm)
        print(f"[AudioProcessing] Clean chunk written: {output_path}")

    return y


def save_waveform(y: np.ndarray, path: str, sample_rate: int = 16000) -> str:
    """Save a float32 waveform as 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pcm = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
    wav_write(path, sample_rate, pcm)
    return path
