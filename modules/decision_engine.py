"""
Module 9: Multimodal Decision Engine
- Combines outputs from face emotion, voice emotion, and text classification.
- Applies rule-based logic to determine the final behavioral classification.

Decision Logic:
    harassment    → text is 'harassment' or 'abusive' AND (face is 'angry' OR voice is 'angry tone')
    stress        → text is 'normal' BUT face is 'angry'/'sad' OR voice is 'angry tone'
    normal        → all signals calm/normal
"""


def make_decision(
    face_emotion: str,
    voice_emotion: str,
    text_prediction: str
) -> str:
    """
    Combine modality predictions into a final behavioral label.

    Args:
        face_emotion:    One of 'angry', 'sad', 'happy', 'neutral'.
        voice_emotion:   One of 'angry tone', 'calm tone'.
        text_prediction: One of 'harassment', 'abusive', 'normal'.

    Returns:
        Final label: 'harassment', 'stress', or 'normal behaviour'.
    """
    face_angry  = face_emotion in ("angry", "sad")
    voice_angry = voice_emotion == "angry tone"
    text_bad    = text_prediction in ("harassment", "abusive")

    if text_bad and (face_angry or voice_angry):
        result = "harassment"
    elif not text_bad and (face_angry or voice_angry):
        result = "stress"
    else:
        result = "normal behaviour"

    print(
        f"[Decision] face={face_emotion} | voice={voice_emotion} | "
        f"text={text_prediction} => {result}"
    )
    return result


if __name__ == "__main__":
    # Quick sanity tests
    cases = [
        ("angry", "angry tone", "harassment"),
        ("neutral", "calm tone", "normal"),
        ("angry", "calm tone", "normal"),
        ("sad", "calm tone", "normal"),
        ("angry", "angry tone", "normal"),
        ("happy", "calm tone", "abusive"),
    ]
    for face, voice, text in cases:
        decision = make_decision(face, voice, text)
        print(f"  [{face} | {voice} | {text}] → {decision}")
