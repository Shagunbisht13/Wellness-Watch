"""
Module 1 — Device Input Module
==============================
Purpose:
    Capture real-time data from hardware (USB webcam, USB microphone).

Technologies:
    Python, OpenCV (cv2.VideoCapture), sounddevice (or PyAudio-compatible).

Output:
    Raw video stream (BGR frames) and raw audio stream (numpy blocks).

Public API
----------
connect_webcam()       → cv2.VideoCapture
connect_microphone()   → (sounddevice.InputStream, queue.Queue)
start_capture()        → generator yielding (frame, audio_chunk) tuples
"""

import queue
import threading
import time
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
import sounddevice as sd


DEFAULT_SAMPLE_RATE: int = 16_000
DEFAULT_CHANNELS: int = 1
DEFAULT_AUDIO_DTYPE: str = "int16"
DEFAULT_BLOCKSIZE: int = 1024
DEFAULT_FPS: int = 20
DEFAULT_FRAME_WIDTH: int = 640
DEFAULT_FRAME_HEIGHT: int = 480


def _scan_cameras(max_index: int = 4) -> list[int]:
    """Return a list of working camera indices (0 … max_index-1)."""
    working = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                working.append(idx)
        cap.release()
    return working


def connect_webcam(
    camera_index: Optional[int] = None,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
) -> cv2.VideoCapture:
    """
    Detect and connect to a USB webcam.

    Prefers external USB (index > 0) over built-in (index 0).
    Returns an opened, configured cv2.VideoCapture.
    """
    if camera_index is None:
        available = _scan_cameras()
        if not available:
            raise RuntimeError(
                "[DeviceInput] No camera device found. "
                "Connect a USB webcam and retry."
            )
        usb = [i for i in available if i > 0]
        camera_index = usb[0] if usb else available[0]
        label = "USB" if usb else "built-in/default"
        print(f"[DeviceInput] {label} camera detected at index {camera_index}")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"[DeviceInput] Failed to open camera at index {camera_index}. "
            "Check that the device is connected and not in use by another app."
        )
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(
        f"[DeviceInput] Webcam opened — index={camera_index}, "
        f"fps={fps}, resolution={width}x{height}"
    )
    return cap


def _detect_mic_device() -> Optional[int]:
    """Prefer a USB input device, else system default, else first input."""
    try:
        devices = sd.query_devices()
        input_devs: list[Tuple[int, str]] = [
            (i, d["name"])
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

        print("[DeviceInput] Available audio input devices:")
        for idx, name in input_devs:
            print(f"              [{idx}] {name}")

        for idx, name in input_devs:
            if "usb" in name.lower():
                print(f"[DeviceInput] USB microphone selected: {name} (index {idx})")
                return idx

        default_device = sd.default.device
        if isinstance(default_device, tuple):
            default_idx = default_device[0]
        else:
            default_idx = default_device

        if default_idx != -1 and default_idx is not None and default_idx < len(devices):
            print(
                f"[DeviceInput] No USB mic found. Using system default: "
                f"{devices[default_idx]['name']} (index {default_idx})"
            )
            return default_idx

        if input_devs:
            idx, name = input_devs[0]
            print(
                f"[DeviceInput] Falling back to first available input: "
                f"{name} (index {idx})"
            )
            return idx

    except Exception as exc:
        print(f"[DeviceInput] Warning — error querying audio devices: {exc}")

    print("[DeviceInput] Using sounddevice default device (None).")
    return None


def connect_microphone(
    device: Optional[int] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    dtype: str = DEFAULT_AUDIO_DTYPE,
    blocksize: int = DEFAULT_BLOCKSIZE,
    audio_queue: Optional[queue.Queue] = None,
) -> Tuple[sd.InputStream, queue.Queue]:
    """
    Connect to the USB microphone via a sounddevice InputStream.

    The stream is configured with a callback that pushes blocks into
    *audio_queue*. Call ``stream.start()`` when ready to receive data.
    """
    if device is None:
        device = _detect_mic_device()

    if audio_queue is None:
        audio_queue = queue.Queue()

    def _audio_callback(indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"[DeviceInput] Audio callback status: {status}")
        audio_queue.put(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            device=device,
            blocksize=blocksize,
            callback=_audio_callback,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[DeviceInput] Failed to open microphone (device={device}): {exc}\n"
            "Make sure a microphone is connected and not in use by another app."
        ) from exc

    print(
        f"[DeviceInput] Microphone stream created — device={device}, "
        f"sample_rate={sample_rate} Hz, channels={channels}, blocksize={blocksize}"
    )
    return stream, audio_queue


def start_capture(
    camera_index: Optional[int] = None,
    mic_device: Optional[int] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    audio_dtype: str = DEFAULT_AUDIO_DTYPE,
    blocksize: int = DEFAULT_BLOCKSIZE,
    fps: int = DEFAULT_FPS,
) -> Generator[Tuple[Optional[np.ndarray], Optional[np.ndarray]], None, None]:
    """
    Start simultaneous live capture of webcam and microphone.

    Yields (frame, audio_chunk) where either may be None on a given iteration.
    Resources are released when the generator exits.
    """
    cap = connect_webcam(camera_index=camera_index, fps=fps)
    audio_q: queue.Queue = queue.Queue()
    mic_stream, audio_q = connect_microphone(
        device=mic_device,
        sample_rate=sample_rate,
        channels=channels,
        dtype=audio_dtype,
        blocksize=blocksize,
        audio_queue=audio_q,
    )

    latest_frame: Optional[np.ndarray] = None
    frame_lock = threading.Lock()
    video_running = threading.Event()
    video_running.set()

    def _video_loop():
        nonlocal latest_frame
        while video_running.is_set():
            ret, frame = cap.read()
            if ret:
                with frame_lock:
                    latest_frame = frame
            else:
                time.sleep(0.01)

    video_thread = threading.Thread(target=_video_loop, daemon=True, name="VideoCapture")
    video_thread.start()
    mic_stream.start()
    print("[DeviceInput] Both webcam and microphone streams are LIVE.")

    try:
        while True:
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None

            audio_chunk = None
            while True:
                try:
                    audio_chunk = audio_q.get_nowait()
                except queue.Empty:
                    break

            yield frame, audio_chunk
            time.sleep(0.001)

    finally:
        print("[DeviceInput] Shutting down capture streams …")
        video_running.clear()
        video_thread.join(timeout=2.0)
        cap.release()
        mic_stream.stop()
        mic_stream.close()
        print("[DeviceInput] All resources released cleanly.")
