"""
Module 8 (Voice): Voice / Speech Emotion Detection
- Extracts acoustic features from an audio file using Librosa:
    MFCC (timbre/phonetic patterns)
    Pitch (F0 fundamental frequency)
    RMS Energy (loudness)
- Applies rule-based thresholds to classify into:
    'angry tone'  — high energy, high pitch variability
    'calm tone'   — baseline / soft speech
- Falls back to 'calm tone' on any error.
"""

import numpy as np


def _load_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        print("[VoiceEmotion] librosa not installed — pip install librosa")
        return None


def predict_voice_emotion(wav_path: str) -> str:
    """
    Predict emotional tone from a WAV audio file.

    Args:
        wav_path: Path to WAV file.

    Returns:
        'angry tone' or 'calm tone'.
    """
    librosa = _load_librosa()
    if librosa is None:
        return _fallback_scipy(wav_path)

    try:
        # Load audio at original sample rate
        y, sr = librosa.load(wav_path, sr=None, mono=True)

        if len(y) == 0:
            return "calm tone"

        # ── Feature 1: RMS Energy ─────────────────────────────────────────
        rms = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms))

        # ── Feature 2: Pitch variability (using harmonic component) ──────
        y_harmonic, _ = librosa.effects.hpss(y)
        pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

        # ── Feature 3: MFCC energy (spectral shape) ───────────────────────
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(np.abs(mfccs)))

        print(
            f"[VoiceEmotion] RMS={mean_rms:.4f} | PitchStd={pitch_std:.1f} | MFCC={mfcc_mean:.2f}"
        )

        # ── Decision thresholds (tuned for typical speech patterns) ───────
        # High energy OR high pitch variability → angry/aggressive tone
        if mean_rms > 0.05 or pitch_std > 150:
            emotion = "angry tone"
        else:
            emotion = "calm tone"

        print(f"[VoiceEmotion] Detected: {emotion}")
        return emotion

    except Exception as e:
        print(f"[VoiceEmotion] Error: {e} — returning 'calm tone'")
        return "calm tone"


def _fallback_scipy(wav_path: str) -> str:
    """
    Fallback when librosa is not installed.
    Uses scipy to measure RMS energy only.
    """
    try:
        from scipy.io.wavfile import read
        rate, data = read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32)
        rms = float(np.sqrt(np.mean(data ** 2)))
        max_val = 32768.0  # int16 max
        normalized_rms = rms / max_val
        print(f"[VoiceEmotion][Fallback] Normalized RMS={normalized_rms:.4f}")
        return "angry tone" if normalized_rms > 0.05 else "calm tone"
    except Exception as e:
        print(f"[VoiceEmotion][Fallback] Error: {e}")
        return "calm tone"


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "temp/audio/test.wav"
    result = predict_voice_emotion(path)
    print(f"Voice Emotion: {result}")
