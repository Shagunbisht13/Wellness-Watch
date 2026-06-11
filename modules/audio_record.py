import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000

print("Start speaking loudly after Enter...")
input()

audio = sd.rec(
    int(10 * fs),
    samplerate=fs,
    channels=1,
    dtype='int16'
)

sd.wait()

write("temp/audio/test.wav", fs, audio)

print("Saved successfully")