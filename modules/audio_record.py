import os
import time
import queue
import threading
import argparse
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, dtype='int16'):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device_index = self.detect_device()
        self.recording = False
        self.thread = None

    def detect_device(self):
        """
        Detects audio input devices.
        Looks for USB microphones first. Falls back to default device.
        """
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    input_devices.append((i, dev['name']))
            
            print("Available Input Devices:")
            for idx, name in input_devices:
                print(f"  [{idx}]: {name}")
            
            # Look for USB device
            for idx, name in input_devices:
                if 'usb' in name.lower():
                    print(f"\nSelecting USB microphone: {name} (index {idx})")
                    return idx
            
            # Fallback to default input device
            default_device = sd.default.device[0]
            if default_device != -1:
                default_name = devices[default_device]['name']
                print(f"\nUSB microphone not found. Falling back to default input device: {default_name} (index {default_device})")
                return default_device
            
            # If default_device is invalid, try picking the first available input device
            if input_devices:
                print(f"\nDefault device invalid. Selecting first available input device: {input_devices[0][1]} (index {input_devices[0][0]})")
                return input_devices[0][0]
                
        except Exception as e:
            print(f"Error querying audio devices: {e}")
        
        print("\nUsing system default device (None)")
        return None  # sounddevice uses default if device is None

    def record_fixed(self, duration_seconds, output_path):
        """
        Record a fixed-duration audio clip and save to output_path.
        Blocks until recording is finished.
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        num_frames = int(duration_seconds * self.sample_rate)
        print(f"Recording {duration_seconds} seconds of audio on device {self.device_index}...")
        
        audio_data = sd.rec(
            num_frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=self.device_index
        )
        sd.wait()
        
        write(output_path, self.sample_rate, audio_data)
        print(f"Recording saved to {output_path}")
        return audio_data

    def start_continuous(self, output_dir, chunk_duration=5):
        """
        Start continuous background recording, writing chunks to files.
        """
        if self.recording:
            print("Already recording.")
            return
        
        self.recording = True
        self.thread = threading.Thread(
            target=self._continuous_record_loop,
            args=(output_dir, chunk_duration),
            daemon=True
        )
        self.thread.start()
        print(f"Started continuous background recording saving to {output_dir}")

    def stop_continuous(self):
        """
        Stop continuous background recording.
        """
        if not self.recording:
            print("Not recording.")
            return
        
        self.recording = False
        if self.thread:
            self.thread.join()
            self.thread = None
        print("Stopped continuous background recording.")

    def _continuous_record_loop(self, output_dir, chunk_duration):
        os.makedirs(output_dir, exist_ok=True)
        q = queue.Queue()
        
        def callback(indata, frames, time_info, status):
            if status:
                print(status, flush=True)
            q.put(indata.copy())
            
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device_index,
                callback=callback
            ):
                chunk_index = 0
                frames_per_chunk = int(chunk_duration * self.sample_rate)
                buffer = []
                buffered_frames = 0
                
                while self.recording:
                    try:
                        # Non-blocking fetch of audio blocks
                        data = q.get(timeout=0.1)
                        buffer.append(data)
                        buffered_frames += len(data)
                        
                        if buffered_frames >= frames_per_chunk:
                            # Concatenate and save the chunk
                            all_data = np.concatenate(buffer, axis=0)
                            chunk_data = all_data[:frames_per_chunk]
                            # Start next buffer with leftovers
                            leftovers = all_data[frames_per_chunk:]
                            buffer = [leftovers] if len(leftovers) > 0 else []
                            buffered_frames = len(leftovers)
                            
                            output_path = os.path.join(output_dir, f"chunk_{chunk_index}.wav")
                            write(output_path, self.sample_rate, chunk_data)
                            print(f"Saved continuous chunk: {output_path}")
                            chunk_index += 1
                            
                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"Error in continuous recording loop: {e}")
            self.recording = False

def list_devices():
    """Helper function to list only available audio input devices."""
    try:
        devices = sd.query_devices()
        print("Available Input Devices:")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}]: {dev['name']} (Max Inputs: {dev['max_input_channels']})")
    except Exception as e:
        print(f"Error querying audio devices: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Device Input Module - Audio Recorder")
    parser.add_argument("--mode", choices=["fixed", "continuous"], default="fixed", help="Recording mode (fixed duration or continuous background chunks)")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds for fixed recording")
    parser.add_argument("--chunk-duration", type=float, default=5.0, help="Duration in seconds for each chunk in continuous recording")
    parser.add_argument("--output", type=str, default="temp/audio/test.wav", help="Output file path for fixed recording")
    parser.add_argument("--output-dir", type=str, default="temp/audio/chunks", help="Output directory for continuous chunks")
    parser.add_argument("--list-devices", action="store_true", help="List all available input devices and exit")
    
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
    elif args.mode == "fixed":
        recorder = AudioRecorder()
        recorder.record_fixed(args.duration, args.output)
    elif args.mode == "continuous":
        recorder = AudioRecorder()
        recorder.start_continuous(args.output_dir, args.chunk_duration)
        print("Recording continuously. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping continuous recording...")
            recorder.stop_continuous()