"""
Module 8c — Text Harassment Classification (BERT)
Purpose: Classify sentence meaning → harassment / abusive / normal.
Technologies: HuggingFace Transformers BERT (unitary/toxic-bert).
              BERT is used instead of TF-IDF because it understands
              sentence meaning and context.

Output: 'harassment', 'abusive', or 'normal'.
"""

import re

from modules.preprocessing import preprocess_text

_classifier = None

_HARASSMENT_LABELS = {"toxic", "severe_toxic", "threat", "insult"}
_ABUSIVE_LABELS = {"obscene", "identity_hate"}

_HARASSMENT_KEYWORDS = {
    "idiot", "stupid", "shut up", "hate you", "loser", "moron",
    "dumb", "kill", "die", "ugly", "worthless", "pathetic",
    "harassment", "abusive", "threat", "insult", "attack",
}


def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline as hf_pipeline
            print("[TextHarassment] Loading BERT toxic classifier…")
            _classifier = hf_pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            print("[TextHarassment] Model loaded successfully.")
        except Exception as e:
            print(f"[TextHarassment] Failed to load model: {e}")
            _classifier = None
    return _classifier


def classify_text(text: str) -> str:
    """
    Classify a text sentence for harassment content using BERT.

    Module 5 cleaning is applied (no stopword removal) before inference.
    """
    if not text or not text.strip():
        return "normal"

    # Light BERT-ready clean — keep stopwords / full context for the model
    cleaned = preprocess_text(text)
    # toxic-bert benefits from original casing slightly; prefer original text
    # but ensure empty after cleaning means skip
    model_input = text.strip() if text.strip() else cleaned

    classifier = _get_classifier()
    if classifier is None:
        return _keyword_fallback(model_input)

    try:
        results = classifier(model_input)
        label_scores = {r["label"].lower(): r["score"] for r in results[0]}
        print(f"[TextHarassment] Scores: {label_scores}")

        threshold = 0.5
        for label in _HARASSMENT_LABELS:
            if label_scores.get(label, 0) >= threshold:
                print(f"[TextHarassment] => harassment (triggered by '{label}')")
                return "harassment"
        for label in _ABUSIVE_LABELS:
            if label_scores.get(label, 0) >= threshold:
                print(f"[TextHarassment] => abusive (triggered by '{label}')")
                return "abusive"

        print("[TextHarassment] => normal")
        return "normal"

    except Exception as e:
        print(f"[TextHarassment] Error: {e} — using keyword fallback")
        return _keyword_fallback(model_input)


def _keyword_fallback(text: str) -> str:
    clean = re.sub(r"[^\w\s]", "", text.lower())
    for keyword in _HARASSMENT_KEYWORDS:
        if keyword in clean:
            print(f"[TextHarassment][Fallback] Keyword match: '{keyword}'")
            return "harassment"
    return "normal"
