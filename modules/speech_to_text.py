import speech_recognition as sr

recognizer = sr.Recognizer()

try:
    with sr.AudioFile("temp/audio/test.wav") as source:
        audio = recognizer.record(source)

    text = recognizer.recognize_google(
    audio,
    language="en-IN"
)

    print("Recognized Text:")
    print(text)

except Exception as e:
    print("Actual Error:")
    print(repr(e))