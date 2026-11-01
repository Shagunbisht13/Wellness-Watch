import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

AUDIO_TEMP_PATH = os.path.join("Temp", "audio", "live_chunk.wav")

from modules.pretrained_models import predict_speech_emotion
from modules.feature_extraction import extract_audio_features
from modules.speech_to_text import transcribe_audio

print("1. predict_speech_emotion")
res1 = predict_speech_emotion(AUDIO_TEMP_PATH)
print("res1", res1)

print("2. extract_audio_features")
res2 = extract_audio_features(AUDIO_TEMP_PATH)
print("res2", type(res2))

print("3. transcribe_audio")
res3 = transcribe_audio(AUDIO_TEMP_PATH)
print("res3", res3)
