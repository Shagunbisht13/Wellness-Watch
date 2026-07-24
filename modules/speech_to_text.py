"""
Module 6 — Speech-to-Text Module
================================
Purpose:
    Convert an audio chunk into a text sentence for harassment analysis.

Technologies:
    SpeechRecognition (Google Web Speech API). Whisper or wav2vec can be
    swapped in later behind the same ``transcribe_audio`` interface.

Flow:
    audio chunk / WAV → speech model → text output

Output:
    Text sentence (empty string on failure / unclear speech).
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from modules.audio_processing import save_waveform

AudioInput = Union[str, np.ndarray]


def transcribe_audio(
    audio: AudioInput,
    language: str = "en-IN",
    sample_rate: int = 16000,
    temp_path: str = "Temp/audio/stt_chunk.wav",
) -> str:
    """
    Transcribe a WAV path or float/int waveform to text.

    Returns an empty string if recognition fails.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        print("[STT] SpeechRecognition not installed.")
        return ""

    wav_path: Optional[str] = None
    cleanup = False

    if isinstance(audio, str):
        wav_path = audio
    else:
        os.makedirs(os.path.dirname(os.path.abspath(temp_path)) or ".", exist_ok=True)
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if np.max(np.abs(y)) > 1.5:
            y = y / (np.max(np.abs(y)) + 1e-8)
        save_waveform(y, temp_path, sample_rate=sample_rate)
        wav_path = temp_path
        cleanup = False  # keep for debugging / proof pipeline

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            recorded = recognizer.record(source)

        text = recognizer.recognize_google(recorded, language=language)
        print(f"[STT] Transcribed: {text!r}")
        return text

    except sr.UnknownValueError:
        print("[STT] Could not understand audio — returning empty string.")
        return ""
    except sr.RequestError as e:
        print(f"[STT] Google Speech API error: {e} — returning empty string.")
        return ""
    except FileNotFoundError:
        print(f"[STT] Audio file not found: {wav_path}")
        return ""
    except Exception as e:
        print(f"[STT] Unexpected error: {e}")
        return ""
    finally:
        if cleanup and wav_path and os.path.isfile(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Temp/audio/test.wav"
    print(f"Recognized Text: {transcribe_audio(path)}")
