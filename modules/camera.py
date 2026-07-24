""" Backward-compatible shim — use ``modules.video_processing``. """
from modules.video_processing import FACE_CASCADE, detect_and_crop_face  # noqa: F401

# Legacy VideoStream kept for older scripts; prefer StreamManager (Module 2).
import threading
from typing import Optional

import cv2
import numpy as np

from modules.device_input import connect_webcam


class VideoStream:
    """Deprecated: prefer ``modules.realtime_stream.StreamManager``."""

    def __init__(self, camera_index=None):
        self.cap = connect_webcam(camera_index=camera_index)
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.cap.release()
