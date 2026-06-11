"""
Module 8 (Text): Text Harassment Classification using BERT
- Uses Hugging Face's 'unitary/toxic-bert' pipeline to classify text.
- Maps model labels to system-level categories:
    'toxic' / 'severe_toxic' / 'threat' / 'insult' → 'harassment'
    'obscene' / 'identity_hate'                     → 'abusive'
    everything else                                  → 'normal'
- Falls back to keyword-based heuristic if HF model is unavailable.
"""

import re

# ── Lazy pipeline loading ─────────────────────────────────────────────────────
_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline as hf_pipeline
            print("[TextClassifier] Loading BERT toxic classifier (first run may download weights)...")
            _classifier = hf_pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                top_k=None,          # return all labels with scores
                truncation=True,
                max_length=512,
            )
            print("[TextClassifier] Model loaded successfully.")
        except Exception as e:
            print(f"[TextClassifier] Failed to load model: {e}")
            _classifier = None
    return _classifier


# Labels from unitary/toxic-bert that map to 'harassment'
_HARASSMENT_LABELS = {"toxic", "severe_toxic", "threat", "insult"}
_ABUSIVE_LABELS    = {"obscene", "identity_hate"}

# Simple keyword fallback list
_HARASSMENT_KEYWORDS = {
    "idiot", "stupid", "shut up", "hate you", "loser", "moron",
    "dumb", "kill", "die", "ugly", "worthless", "pathetic",
    "harassment", "abusive", "threat", "insult", "attack",
}


def classify_text(text: str) -> str:
    """
    Classify a text sentence for harassment content.

    Args:
        text: Input sentence string.

    Returns:
        'harassment', 'abusive', or 'normal'.
    """
    if not text or not text.strip():
        return "normal"

    classifier = _get_classifier()
    if classifier is None:
        return _keyword_fallback(text)

    try:
        results = classifier(text)
        # results is a list of [{label, score}, ...] for all labels
        label_scores = {r["label"].lower(): r["score"] for r in results[0]}
        print(f"[TextClassifier] Scores: {label_scores}")

        # Check if any harassment label exceeds threshold
        THRESHOLD = 0.5
        for label in _HARASSMENT_LABELS:
            if label_scores.get(label, 0) >= THRESHOLD:
                print(f"[TextClassifier] => harassment (triggered by '{label}')")
                return "harassment"
        for label in _ABUSIVE_LABELS:
            if label_scores.get(label, 0) >= THRESHOLD:
                print(f"[TextClassifier] => abusive (triggered by '{label}')")
                return "abusive"

        print("[TextClassifier] => normal")
        return "normal"

    except Exception as e:
        print(f"[TextClassifier] Error: {e} - using keyword fallback")
        return _keyword_fallback(text)


def _keyword_fallback(text: str) -> str:
    """Simple keyword-based fallback classifier."""
    clean = re.sub(r"[^\w\s]", "", text.lower())
    for keyword in _HARASSMENT_KEYWORDS:
        if keyword in clean:
            print(f"[TextClassifier][Fallback] Keyword match: '{keyword}'")
            return "harassment"
    return "normal"


if __name__ == "__main__":
    samples = [
        "shut up you idiot",
        "good morning everyone",
        "I will hurt you if you don't stop",
        "you are doing great today",
    ]
    for s in samples:
        print(f"  '{s}' → {classify_text(s)}")