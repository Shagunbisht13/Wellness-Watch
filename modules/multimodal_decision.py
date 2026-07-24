"""
Module 9 — Multimodal Decision Module
=====================================
Purpose:
    Fuse face emotion, speech tone, and text harassment outputs into one
    behavioural verdict.

Technologies:
    Python rule-based fusion (extensible to learned fusion later).

Logic (approved):
    IF face_emotion == angry
       AND text_class == harassment
       AND speech_tone == aggressive (angry tone)
    THEN harassment detected

    Partial concerning signals → stress
    Otherwise → normal behaviour

Output classes:
    harassment / stress / normal behaviour
"""


def make_decision(
    face_emotion: str,
    voice_emotion: str,
    text_prediction: str,
) -> str:
    """
    Combine modality predictions into a final behavioural label.

    Parameters
    ----------
    face_emotion : str
        One of 'angry', 'sad', 'happy', 'neutral'.
    voice_emotion : str
        One of 'angry tone', 'calm tone' (aggressive ≡ angry tone).
    text_prediction : str
        One of 'harassment', 'abusive', 'normal'.

    Returns
    -------
    str
        'harassment', 'stress', or 'normal behaviour'.
    """
    face_angry = face_emotion == "angry"
    voice_aggressive = voice_emotion in ("angry tone", "aggressive")
    text_harassment = text_prediction in ("harassment", "abusive")

    # Spec: all three modalities agree on aggression/harassment
    if face_angry and text_harassment and voice_aggressive:
        result = "harassment"
    # Partial signals (emotion OR toxic text without full triple match)
    elif face_emotion in ("angry", "sad") or voice_aggressive or text_harassment:
        result = "stress"
    else:
        result = "normal behaviour"

    print(
        f"[Decision] face={face_emotion} | voice={voice_emotion} | "
        f"text={text_prediction} => {result}"
    )
    return result


if __name__ == "__main__":
    cases = [
        ("angry", "angry tone", "harassment"),
        ("neutral", "calm tone", "normal"),
        ("angry", "calm tone", "normal"),
        ("sad", "calm tone", "normal"),
        ("angry", "angry tone", "normal"),
        ("happy", "calm tone", "abusive"),
    ]
    for face, voice, text in cases:
        print(f"  [{face} | {voice} | {text}] → {make_decision(face, voice, text)}")
