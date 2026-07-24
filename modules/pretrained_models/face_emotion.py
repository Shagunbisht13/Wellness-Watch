"""
Module 8a — Face Emotion (FER2013 / ViT-CNN)
Purpose: Classify cropped face → happy / sad / angry / neutral.
Technologies: HuggingFace Transformers, PyTorch, OpenCV, Pillow.
Output: Emotion label string.
"""

import numpy as np
import cv2
from PIL import Image

from modules.preprocessing import preprocess_face

_pipeline = None

_LABEL_MAP = {
    "angry": "angry",
    "disgust": "angry",
    "fear": "angry",
    "happy": "happy",
    "sad": "sad",
    "surprise": "happy",
    "neutral": "neutral",
}


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            print("[FaceEmotion] Loading FER2013 face emotion model…")
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


def predict_face_emotion(face_bgr: np.ndarray) -> str:
    """
    Predict emotion from a cropped face BGR image.

    Applies Module 5 image preprocessing (resize 224×224) before inference.
    """
    if face_bgr is None or face_bgr.size == 0:
        return "neutral"

    pipe = _get_pipeline()
    if pipe is None:
        return _heuristic_emotion(face_bgr)

    try:
        processed = preprocess_face(face_bgr, size=(224, 224))
        # HF pipeline expects PIL uint8 RGB
        rgb_u8 = (np.clip(processed, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(rgb_u8)
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
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return "neutral" if brightness > 80 else "sad"
