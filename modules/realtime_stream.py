"""
Module 2 — Real-Time Stream Module
==================================
Purpose:
    Capture webcam and microphone simultaneously using threading.

Technologies:
    Python threading, queue, OpenCV (via Module 1), sounddevice (via Module 1).

Behaviour:
    Thread 1 — continuously capture webcam frames (capture at 15–20 FPS,
               forward for processing at 1 FPS).
    Thread 2 — continuously record audio in 3–5 second chunks.

Output:
    Live frame queue and audio chunk queue (via StreamManager).
"""

import queue
import threading
import time
from typing import Optional

import numpy as np

from modules.device_input import connect_microphone, connect_webcam


class VideoThread(threading.Thread):
    """Capture frames at target FPS; enqueue one processed frame per second."""

    def __init__(
        self,
        cap,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        target_fps: int = 18,
    ):
        super().__init__(daemon=True, name="VideoThread")
        self.cap = cap
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.target_fps = float(max(1, target_fps))

    def run(self) -> None:
        try:
            frame_interval = 1.0 / self.target_fps
            last_forward_time = time.perf_counter()
            latest_frame: Optional[np.ndarray] = None

            while not self.stop_event.is_set():
                start_time = time.perf_counter()
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    latest_frame = frame.copy()

                now = time.perf_counter()
                # Process / forward at 1 FPS (keep freshest frame)
                if now - last_forward_time >= 1.0 and latest_frame is not None:
                    try:
                        self.frame_queue.put(latest_frame, block=False)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(latest_frame, block=False)
                        except queue.Empty:
                            pass
                    last_forward_time = now

                elapsed = time.perf_counter() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.05))
        except Exception as exc:
            print(f"[RealtimeStream] VideoThread error: {exc}")


class AudioThread(threading.Thread):
    """Collect mic blocks and forward 3–5 second audio chunks."""

    def __init__(
        self,
        mic_stream,
        raw_audio_queue: queue.Queue,
        chunk_queue: queue.Queue,
        stop_event: threading.Event,
        sample_rate: int = 16000,
        chunk_duration: float = 4.0,
    ):
        super().__init__(daemon=True, name="AudioThread")
        self.mic_stream = mic_stream
        self.raw_audio_queue = raw_audio_queue
        self.chunk_queue = chunk_queue
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.chunk_duration = max(3.0, min(chunk_duration, 5.0))

    def run(self) -> None:
        try:
            self.mic_stream.start()
            target_frames = int(self.sample_rate * self.chunk_duration)
            audio_blocks = []
            collected_frames = 0

            while not self.stop_event.is_set():
                try:
                    block = self.raw_audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if block is None:
                    continue

                audio_blocks.append(block)
                collected_frames += len(block)

                if collected_frames >= target_frames:
                    chunk = np.concatenate(audio_blocks, axis=0)
                    try:
                        self.chunk_queue.put(chunk, block=False)
                    except queue.Full:
                        try:
                            self.chunk_queue.get_nowait()
                            self.chunk_queue.put(chunk, block=False)
                        except queue.Empty:
                            pass
                    audio_blocks = []
                    collected_frames = 0

            if audio_blocks:
                chunk = np.concatenate(audio_blocks, axis=0)
                try:
                    self.chunk_queue.put(chunk, block=False)
                except queue.Full:
                    pass
        except Exception as exc:
            print(f"[RealtimeStream] AudioThread error: {exc}")
        finally:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass


class StreamManager:
    """
    Manages real-time video and audio streaming threads.

    Output queues:
        frame_queue        — live webcam frames (≈1 FPS for processing)
        audio_chunk_queue  — 3–5 s raw audio chunks
    """

    def __init__(
        self,
        camera_index: Optional[int] = None,
        mic_device: Optional[int] = None,
        frame_fps: int = 18,
        audio_chunk_duration: float = 4.0,
        sample_rate: int = 16000,
        channels: int = 1,
        audio_dtype: str = "int16",
    ):
        self.sample_rate = sample_rate
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.audio_chunk_queue: queue.Queue = queue.Queue(maxsize=4)
        self.raw_audio_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()

        self.cap = connect_webcam(camera_index=camera_index, fps=frame_fps)
        self.mic_stream, self.raw_audio_queue = connect_microphone(
            device=mic_device,
            sample_rate=sample_rate,
            channels=channels,
            dtype=audio_dtype,
            audio_queue=self.raw_audio_queue,
        )

        self.video_thread = VideoThread(
            cap=self.cap,
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
            target_fps=frame_fps,
        )
        self.audio_thread = AudioThread(
            mic_stream=self.mic_stream,
            raw_audio_queue=self.raw_audio_queue,
            chunk_queue=self.audio_chunk_queue,
            stop_event=self.stop_event,
            sample_rate=sample_rate,
            chunk_duration=audio_chunk_duration,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.video_thread.start()
        self.audio_thread.start()
        print("[RealtimeStream] StreamManager started both threads.")

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_audio_chunk(self) -> Optional[np.ndarray]:
        try:
            return self.audio_chunk_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self, timeout: float = 2.0) -> None:
        self.stop_event.set()
        self.video_thread.join(timeout=timeout)
        self.audio_thread.join(timeout=timeout)
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        print("[RealtimeStream] StreamManager stopped and joined threads.")
