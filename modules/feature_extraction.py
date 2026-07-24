"""
Module 7 — Feature Extraction Module
====================================
Purpose:
    Extract modality-specific feature vectors used by (or alongside)
    the pre-trained models in Module 8.

Technologies:
    Text  — HuggingFace BERT tokenizer + model → contextual embeddings
            (understands context/tone, not just keywords; not TF-IDF/BoW)
    Image — lightweight CNN / spatial pooling → facial feature vector
    Audio — Librosa MFCC → speech-pattern feature vector

Output:
    Text embedding, image feature vector, audio MFCC feature vector.
"""

from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np

from modules.preprocessing import preprocess_text

AudioInput = Union[np.ndarray, str]

_bert_tokenizer = None
_bert_model = None


def _get_bert():
    """Lazy-load a BERT base model for contextual embeddings."""
    global _bert_tokenizer, _bert_model
    if _bert_model is not None:
        return _bert_tokenizer, _bert_model
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        name = "bert-base-uncased"
        print("[FeatureExtraction] Loading BERT for text embeddings…")
        _bert_tokenizer = AutoTokenizer.from_pretrained(name)
        _bert_model = AutoModel.from_pretrained(name)
        _bert_model.eval()
        print("[FeatureExtraction] BERT loaded.")
    except Exception as exc:
        print(f"[FeatureExtraction] BERT unavailable: {exc}")
        _bert_tokenizer, _bert_model = None, None
    return _bert_tokenizer, _bert_model


def extract_text_features(text: str) -> np.ndarray:
    """
    BERT contextual embedding (CLS pooled mean) for *text*.

    Stopwords are kept — only Module 5 light cleaning is applied.
    Falls back to a hashed bag-of-char vector if BERT is unavailable
    (never uses TF-IDF for the primary path).
    """
    cleaned = preprocess_text(text)
    if not cleaned:
        return np.zeros(768, dtype=np.float32)

    tokenizer, model = _get_bert()
    if tokenizer is None or model is None:
        return _hash_fallback_embedding(cleaned, dim=768)

    try:
        import torch

        encoded = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**encoded)
            # Mean-pool last hidden state (contextual embeddings)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as exc:
        print(f"[FeatureExtraction] BERT encode error: {exc}")
        return _hash_fallback_embedding(cleaned, dim=768)


def _hash_fallback_embedding(text: str, dim: int = 768) -> np.ndarray:
    """Deterministic non-TF-IDF fallback when BERT weights are missing."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.split():
        idx = hash(token) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def extract_image_features(face_bgr: np.ndarray) -> np.ndarray:
    """
    Facial feature vector via a small CNN-style spatial pyramid of edges
    and intensity (eyes/mouth region proxies without a separate train step).

    When a full CNN backbone is available, this can be replaced with its
    penultimate layer activations.
    """
    if face_bgr is None or face_bgr.size == 0:
        return np.zeros(128, dtype=np.float32)

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gray_f = gray.astype(np.float32) / 255.0

    # Simple "CNN-like" filters: horizontal/vertical Sobel + local means
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    features = []
    for tile in (gray_f, np.abs(gx), np.abs(gy)):
        for i in range(0, 64, 16):
            for j in range(0, 64, 16):
                patch = tile[i : i + 16, j : j + 16]
                features.append(float(np.mean(patch)))
                features.append(float(np.std(patch)))
    vec = np.asarray(features, dtype=np.float32)
    if vec.size < 128:
        vec = np.pad(vec, (0, 128 - vec.size))
    return vec[:128]


def extract_audio_features(
    audio: AudioInput,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Librosa MFCC mean+std vector describing speech patterns."""
    try:
        import librosa
    except ImportError:
        print("[FeatureExtraction] librosa missing — zero MFCC vector")
        return np.zeros(n_mfcc * 2, dtype=np.float32)

    try:
        if isinstance(audio, str):
            y, sr = librosa.load(audio, sr=sample_rate, mono=True)
        else:
            y = np.asarray(audio, dtype=np.float32)
            if y.ndim > 1:
                y = y.mean(axis=1)
            sr = sample_rate
            if np.max(np.abs(y)) > 1.5:
                y = y / (np.max(np.abs(y)) + 1e-8)

        if y.size == 0:
            return np.zeros(n_mfcc * 2, dtype=np.float32)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mean = np.mean(mfcc, axis=1)
        std = np.std(mfcc, axis=1)
        return np.concatenate([mean, std]).astype(np.float32)
    except Exception as exc:
        print(f"[FeatureExtraction] MFCC error: {exc}")
        return np.zeros(n_mfcc * 2, dtype=np.float32)


def extract_all_features(
    face_bgr: Optional[np.ndarray] = None,
    audio: Optional[AudioInput] = None,
    text: Optional[str] = None,
) -> dict:
    """Extract available modality features in one call."""
    out = {}
    if face_bgr is not None:
        out["image"] = extract_image_features(face_bgr)
    if audio is not None:
        out["audio"] = extract_audio_features(audio)
    if text is not None:
        out["text"] = extract_text_features(text)
    return out
