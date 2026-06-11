"""
Module 6: Speech-to-Text
- Transcribes a WAV audio file into text using Google Speech Recognition API.
- Wraps SpeechRecognition library for easy integration.
- Falls back gracefully on recognition errors (network issues, unclear speech, etc).
"""

import speech_recognition as sr


def transcribe_audio(wav_path: str, language: str = "en-IN") -> str:
    """
    Transcribe a WAV audio file to text.

    Args:
        wav_path: Absolute or relative path to the WAV file.
        language: BCP-47 language code for recognition. Defaults to 'en-IN'.

    Returns:
        Transcribed text string. Empty string if recognition fails or audio is unclear.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            # Adjust for ambient noise in first 0.5 seconds of the clip
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language=language)
        print(f"[STT] Transcribed: {text!r}")
        return text

    except sr.UnknownValueError:
        print("[STT] Could not understand audio — returning empty string.")
        return ""
    except sr.RequestError as e:
        print(f"[STT] Google Speech API error: {e} — returning empty string.")
        return ""
    except FileNotFoundError:
        print(f"[STT] Audio file not found: {wav_path}")
        return ""
    except Exception as e:
        print(f"[STT] Unexpected error: {e}")
        return ""


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "temp/audio/test.wav"
    result = transcribe_audio(path)
    print(f"Recognized Text: {result}")