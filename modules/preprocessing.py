"""
Module 5 — Preprocessing Module
===============================
Purpose:
    Prepare face images, audio waveforms, and transcribed text for model input.

Technologies:
    Image — OpenCV / NumPy (resize, pixel normalization)
    Audio — Librosa (noise removal, silence trimming, normalization)
    Text  — NLTK-style cleaning + HuggingFace tokenizer prep
            (lowercase, strip punctuation, tokenize).
            NOTE: Stopword removal is intentionally skipped — BERT needs
            full sentence context.

Output:
    Processed face image, processed audio signal, and clean text string.
"""

from __future__ import annotations

import re
import string
from typing import Optional, Tuple, Union

import cv2
import numpy as np

AudioInput = Union[np.ndarray, str]


def preprocess_face(
    face_bgr: np.ndarray,
    size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Resize face to *size* and normalize pixel values to [0, 1] float32.

    Default size 224×224 suits ViT/CNN models; pass (48, 48) for FER-style CNNs.
    """
    if face_bgr is None or face_bgr.size == 0:
        return np.zeros((*size, 3), dtype=np.float32)

    resized = cv2.resize(face_bgr, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)


def preprocess_audio(
    audio: AudioInput,
    sample_rate: int = 16000,
    top_db: float = 25.0,
) -> np.ndarray:
    """
    Noise-aware silence trim + amplitude normalization for a waveform or WAV path.
    """
    try:
        import librosa
    except ImportError:
        print("[Preprocessing] librosa missing — returning raw audio")
        if isinstance(audio, str):
            from scipy.io.wavfile import read as wav_read
            _, data = wav_read(audio)
            y = data.astype(np.float32)
            if y.ndim > 1:
                y = y.mean(axis=1)
            peak = np.max(np.abs(y)) or 1.0
            return (y / peak).astype(np.float32)
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        peak = np.max(np.abs(y)) or 1.0
        return (y / peak).astype(np.float32)

    if isinstance(audio, str):
        y, sr = librosa.load(audio, sr=sample_rate, mono=True)
    else:
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        sr = sample_rate
        # Rescale int-like ranges
        if np.max(np.abs(y)) > 1.5:
            y = y / (np.max(np.abs(y)) + 1e-8)

    try:
        y, _ = librosa.effects.trim(y, top_db=top_db)
    except Exception:
        pass

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = (y / peak * 0.95).astype(np.float32)
    return y.astype(np.float32)


def preprocess_text(text: str, lowercase: bool = True) -> str:
    """
    BERT-ready text cleaning.

    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    - Does NOT remove stopwords (BERT needs full context)
    """
    if not text:
        return ""

    cleaned = text.strip()
    if lowercase:
        cleaned = cleaned.lower()

    # Remove punctuation but keep word tokens / spaces
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_text(text: str) -> list[str]:
    """
    Simple whitespace tokenization after preprocess_text.
    For model inference, Module 7 / 8 use the HuggingFace BERT tokenizer.
    """
    cleaned = preprocess_text(text)
    return cleaned.split() if cleaned else []


def preprocess_for_pipeline(
    face_bgr: Optional[np.ndarray] = None,
    audio: Optional[AudioInput] = None,
    text: Optional[str] = None,
    face_size: Tuple[int, int] = (224, 224),
    sample_rate: int = 16000,
) -> dict:
    """Convenience batch preprocessor for the orchestrator."""
    result = {}
    if face_bgr is not None:
        result["face"] = preprocess_face(face_bgr, size=face_size)
        result["face_bgr_resized"] = cv2.resize(
            face_bgr, face_size, interpolation=cv2.INTER_AREA
        )
    if audio is not None:
        result["audio"] = preprocess_audio(audio, sample_rate=sample_rate)
    if text is not None:
        result["text"] = preprocess_text(text)
        result["tokens"] = tokenize_text(text)
    return result
