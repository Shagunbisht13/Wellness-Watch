""" Backward-compatible shim — use ``modules.audio_processing``. """
from modules.audio_processing import process_audio_chunk, save_waveform  # noqa: F401

# Legacy AudioRecorder for scripts that still call record_fixed().
import os

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from modules.device_input import _detect_mic_device


class AudioRecorder:
    """Deprecated: prefer StreamManager + process_audio_chunk."""

    def __init__(self, sample_rate=16000, channels=1, dtype="int16"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device_index = _detect_mic_device()

    def record_fixed(self, duration_seconds, output_path):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        num_frames = int(duration_seconds * self.sample_rate)
        audio_data = sd.rec(
            num_frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=self.device_index,
        )
        sd.wait()
        write(output_path, self.sample_rate, audio_data)
        return audio_data
