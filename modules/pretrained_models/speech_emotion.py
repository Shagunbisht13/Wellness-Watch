"""
Module 8b — Speech Emotion (RAVDESS-oriented labels)
Purpose: Detect speech tone → angry tone / calm tone.
Technologies: Librosa MFCC/pitch/RMS (CNN/LSTM SER weights can replace this).
Output: 'angry tone' or 'calm tone'.

NOTE (gap flag): A full RAVDESS-trained LSTM/CNN checkpoint is not bundled.
This module preserves the approved output contract using acoustic features
from Module 7 until a trained SER model is added under models/.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from modules.feature_extraction import extract_audio_features
from modules.preprocessing import preprocess_audio

AudioInput = Union[str, np.ndarray]


def predict_speech_emotion(audio: AudioInput, sample_rate: int = 16000) -> str:
    """
    Predict emotional tone from a WAV path or waveform.

    Returns 'angry tone' or 'calm tone'.
    """
    try:
        y = preprocess_audio(audio, sample_rate=sample_rate)
        if y.size == 0:
            return "calm tone"

        import librosa

        rms = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms))

        y_harmonic, _ = librosa.effects.hpss(y)
        pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sample_rate)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

        mfcc_vec = extract_audio_features(y, sample_rate=sample_rate)
        mfcc_energy = float(np.mean(np.abs(mfcc_vec[:13]))) if mfcc_vec.size else 0.0

        print(
            f"[SpeechEmotion] RMS={mean_rms:.4f} | PitchStd={pitch_std:.1f} | "
            f"MFCC={mfcc_energy:.2f}"
        )

        if mean_rms > 0.05 or pitch_std > 150:
            emotion = "angry tone"
        else:
            emotion = "calm tone"

        print(f"[SpeechEmotion] Detected: {emotion}")
        return emotion

    except Exception as e:
        print(f"[SpeechEmotion] Error: {e} — returning 'calm tone'")
        return "calm tone"


# Alias matching older Module 8 naming
predict_voice_emotion = predict_speech_emotion
