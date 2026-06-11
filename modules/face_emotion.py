"""
Module 8 (Face): Face Emotion Detection
- Loads a pre-trained Hugging Face image-classification pipeline using
  a lightweight ViT model fine-tuned on FER2013 for facial emotion recognition.
- Classifies a cropped face BGR image into one of: happy, sad, angry, neutral, etc.
- Falls back to 'neutral' if an error occurs to avoid crashing the pipeline.
"""

import numpy as np
import cv2
from PIL import Image

# ── Model loading ──────────────────────────────────────────────────────────────
# We lazy-load the pipeline on first use to avoid slow import times.
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            print("[FaceEmotion] Loading face emotion model (first run may download weights)...")
            _pipeline = hf_pipeline(
                "image-classification",
                model="trpakov/vit-face-expression",
                top_k=1,
            )
            print("[FaceEmotion] Model loaded successfully.")
        except Exception as e:
            print(f"[FaceEmotion] Failed to load Hugging Face model: {e}")
            _pipeline = None
    return _pipeline


# Canonical emotion labels mapped to simplified labels used by decision engine
_LABEL_MAP = {
    "angry":     "angry",
    "disgust":   "angry",
    "fear":      "angry",
    "happy":     "happy",
    "sad":       "sad",
    "surprise":  "happy",
    "neutral":   "neutral",
}


def predict_face_emotion(face_bgr: np.ndarray) -> str:
    """
    Predict the emotion from a cropped face image.

    Args:
        face_bgr: BGR face image as NumPy array (as returned by OpenCV).

    Returns:
        Emotion label string: 'angry', 'happy', 'sad', or 'neutral'.
    """
    if face_bgr is None or face_bgr.size == 0:
        return "neutral"

    pipe = _get_pipeline()
    if pipe is None:
        # Fallback: simple brightness heuristic when model unavailable
        return _heuristic_emotion(face_bgr)

    try:
        # Convert BGR → RGB → PIL Image for the HF pipeline
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        result = pipe(pil_image)
        raw_label = result[0]["label"].lower()
        emotion = _LABEL_MAP.get(raw_label, "neutral")
        score = result[0]["score"]
        print(f"[FaceEmotion] Detected: {emotion} (raw={raw_label}, confidence={score:.2f})")
        return emotion
    except Exception as e:
        print(f"[FaceEmotion] Prediction error: {e} — returning 'neutral'")
        return "neutral"


def _heuristic_emotion(face_bgr: np.ndarray) -> str:
    """
    Simple fallback when model is not available.
    Analyzes overall brightness as a very rough proxy.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return "neutral" if brightness > 80 else "sad"


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        img = cv2.imread(path)
        result = predict_face_emotion(img)
        print(f"Emotion: {result}")
    else:
        print("Usage: python face_emotion.py <path_to_face_image>")
