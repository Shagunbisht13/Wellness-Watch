"""
Module 8 — Pre-trained Model Module
===================================
Purpose:
    Run face-emotion, speech-emotion, and text-harassment classifiers.

Technologies:
    Face Emotion   — CNN / ViT (FER2013), TensorFlow/PyTorch via HuggingFace
    Speech Emotion — acoustic LSTM/CNN-style path (RAVDESS-oriented labels);
                     current implementation uses Librosa features + thresholds
                     with the same angry/calm output contract
    Text Harassment — BERT via HuggingFace Transformers
                      (chosen over TF-IDF for sentence meaning/context)

Output:
    Face: happy / sad / angry / neutral
    Speech: angry tone / calm tone
    Text: harassment / abusive / normal
"""

from modules.pretrained_models.face_emotion import predict_face_emotion
from modules.pretrained_models.speech_emotion import predict_speech_emotion
from modules.pretrained_models.text_harassment import classify_text

# Back-compat alias used by older call sites
predict_voice_emotion = predict_speech_emotion

__all__ = [
    "predict_face_emotion",
    "predict_speech_emotion",
    "predict_voice_emotion",
    "classify_text",
]
